import tempfile
import os
import logging
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoProcessor, AutoModel
from datetime import datetime
import faiss
import pickle
import uuid
from langchain_community.llms import Ollama
import tempfile
import os
import logging
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
import faiss
import pickle
import uuid
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import whisper
from langchain_community.llms import Ollama


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Whisper model for audio transcription
logger.info("Loading Whisper model...")
whisper_model = whisper.load_model("base")
logger.info("Whisper model loaded successfully")

model = Ollama(model="gemma:2b")

logger.info("Loading Minilm-l6 model for text embeddings...")
minilm_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
minilm_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
logger.info("Minilm-l6 model loaded successfully")

# Create data directories if they don't exist
os.makedirs("data/videos", exist_ok=True)
os.makedirs("data/frames", exist_ok=True)
os.makedirs("data/search_index", exist_ok=True)

# FAISS index setup
index_file = "data/search_index/faiss_index.pkl"
metadata_file = "data/search_index/metadata.pkl"

# Initialize or load FAISS index
if os.path.exists(index_file) and os.path.exists(metadata_file):
    logger.info("Loading existing FAISS index...")
    with open(index_file, "rb") as f:
        index = pickle.load(f)
    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)
else:
    logger.info("Creating new FAISS index...")
    # Create a new index - we'll use 512 dimensions for CLIP embeddings
    index = faiss.IndexFlatIP(512)  # Inner product for cosine similarity
    metadata = []

# Function to extract frames from video
def extract_frames(video_path, output_dir, max_frames=20):
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error opening video file: {video_path}")
        return []
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps
    
    # Extract frames at regular intervals
    interval = max(1, frame_count // max_frames)
    frames_info = []
    
    for i in range(0, frame_count, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate timestamp
        timestamp_seconds = i / fps
        timestamp = str(datetime.utcfromtimestamp(timestamp_seconds).strftime('%H:%M:%S'))
        
        # Save frame
        frame_filename = f"{output_dir}/frame_{i:04d}.jpg"
        cv2.imwrite(frame_filename, frame)
        
        frames_info.append({
            "frame_id": i,
            "timestamp": timestamp,
            "timestamp_seconds": timestamp_seconds,
            "path": frame_filename
        })
        
    cap.release()
    return frames_info

# Function to transcribe audio from video
def transcribe_audio(video_path):
    import subprocess
    
    # Extract audio from video
    audio_path = f"{os.path.splitext(video_path)[0]}.wav"
    command = f"ffmpeg -i {video_path} -q:a 0 -map a {audio_path} -y"
    subprocess.call(command, shell=True)
    
    # Transcribe audio
    result = whisper_model.transcribe(audio_path)
    segments = result["segments"]
    
    # Format segments with timestamps
    transcript_segments = []
    for segment in segments:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]
        
        # Format timestamps
        start_timestamp = str(datetime.utcfromtimestamp(start_time).strftime('%H:%M:%S'))
        end_timestamp = str(datetime.utcfromtimestamp(end_time).strftime('%H:%M:%S'))
        
        transcript_segments.append({
            "start": start_timestamp,
            "end": end_timestamp,
            "start_seconds": start_time,
            "end_seconds": end_time,
            "text": text
        })
    
    return transcript_segments

def generate_text_embedding(text):
    inputs = minilm_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = minilm_model(**inputs)
    
    # Get embeddings from the output
    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Use mean of all tokens as embedding
    embedding = embedding / np.linalg.norm(embedding)  # Normalize the embedding
    return embedding[0]  # Return the vector without batch dimension

# Function to generate embeddings for a given image
def generate_image_embedding(image_path):
    from PIL import Image
    
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    
    # Normalize the embedding
    embedding = outputs.cpu().numpy()
    embedding = embedding / np.linalg.norm(embedding)
    return embedding[0]  # Return the vector without batch dimension

# Function to add data to FAISS index
def add_to_index(embedding, meta_info):
    global index, metadata
    
    # Add the embedding to the index
    index.add(np.array([embedding]).astype('float32'))
    
    # Add metadata
    metadata.append(meta_info)
    
    # Save the updated index and metadata
    with open(index_file, "wb") as f:
        pickle.dump(index, f)
    with open(metadata_file, "wb") as f:
        pickle.dump(metadata, f)

# Function to search FAISS index
def search_index(query_embedding, k=5, threshold=0.7):
    if index.ntotal == 0:
        return []
    
    # Search the index
    distances, indices = index.search(np.array([query_embedding]).astype('float32'), k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1 and distances[0][i] > threshold:  # -1 means no match found
            result = {
                "metadata": metadata[idx],
                "score": float(distances[0][i]),
                "content": metadata[idx].get("text", ""),
                "timestamp": metadata[idx].get("timestamp", ""),
                "timestamp_seconds": metadata[idx].get("timestamp_seconds", 0),
                "video_id": metadata[idx].get("video_id", "")
            }
            results.append(result)
    
    # Sort results by score in descending order
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

# Endpoint to process and analyze video
@app.post("/analyze_video")
async def analyze_video(file: UploadFile = File(...)):
    try:
        logger.info(f"Receiving video file: {file.filename}, content_type: {file.content_type}")
        
        # Generate a unique ID for this video
        video_id = str(uuid.uuid4())
        
        # Create a directory for this video's frames
        frames_dir = f"data/frames/{video_id}"
        os.makedirs(frames_dir, exist_ok=True)
        
        # Save the video file
        video_path = f"data/videos/{video_id}_{file.filename}"
        with open(video_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Video saved to: {video_path}")
        
        # Extract frames
        logger.info("Extracting frames from video...")
        frames_info = extract_frames(video_path, frames_dir)
        
        # Transcribe audio
        logger.info("Transcribing audio from video...")
        transcript_segments = transcribe_audio(video_path)
        
        # Generate embeddings and add to index
        logger.info("Generating embeddings for frames and transcript segments...")
        
        # Process frames
        for frame in frames_info:
            try:
                embedding = generate_image_embedding(frame["path"])
                
                # Add to index
                meta_info = {
                    "type": "video_frame",
                    "source": "video_frame",
                    "video_id": video_id,
                    "frame_id": frame["frame_id"],
                    "timestamp": frame["timestamp"],
                    "timestamp_seconds": frame["timestamp_seconds"],
                    "path": frame["path"]
                }
                add_to_index(embedding, meta_info)
            except Exception as e:
                logger.error(f"Error processing frame {frame['frame_id']}: {str(e)}")
        
        # Process transcript segments
        for segment in transcript_segments:
            try:
                embedding = generate_text_embedding(segment["text"])
                
                # Add to index
                meta_info = {
                    "type": "video_transcript",
                    "source": "video_transcript",
                    "video_id": video_id,
                    "text": segment["text"],
                    "timestamp": segment["start"],
                    "timestamp_seconds": segment["start_seconds"],
                    "end_timestamp": segment["end"],
                    "end_timestamp_seconds": segment["end_seconds"]
                }
                add_to_index(embedding, meta_info)
            except Exception as e:
                logger.error(f"Error processing transcript segment: {str(e)}")
        
        # Generate overall video analysis
        all_transcript_text = " ".join([segment["text"] for segment in transcript_segments])
        analysis = f"Video Analysis Summary:\n\n" + \
                  f"Duration: {frames_info[-1]['timestamp_seconds'] - frames_info[0]['timestamp_seconds']:.2f} seconds\n" + \
                  f"Frames Extracted: {len(frames_info)}\n" + \
                  f"Transcript Segments: {len(transcript_segments)}\n\n" + \
                  f"Content Summary:\n{all_transcript_text[:500]}..."
        
        # Create response
        response = {
            "video_id": video_id,
            "filename": file.filename,
            "frames_count": len(frames_info),
            "transcript_segments_count": len(transcript_segments),
            "analysis": analysis,
            "frames": frames_info,
            "transcript": transcript_segments
        }
        
        return JSONResponse(content=response, status_code=200)
    except Exception as e:
        logger.error(f"Error in analyze_video: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Endpoint to search content
@app.post("/search")
async def search(
    query: str = Form(...),
    search_type: str = Form("text"),  # "text" or "image"
    file: Optional[UploadFile] = File(None),
    similarity_threshold: float = Form(0.7)
):
    try:
        query_embedding = None
        
        if search_type == "text":
            # Generate text embedding
            query_embedding = generate_text_embedding(query)
        elif search_type == "image" and file:
            # Save the uploaded image temporarily
            temp_image_path = f"data/temp_{uuid.uuid4()}.jpg"
            with open(temp_image_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Generate image embedding
            query_embedding = generate_image_embedding(temp_image_path)
            
            # Clean up
            os.unlink(temp_image_path)
        else:
            return JSONResponse(content={"error": "Invalid search type or missing file"}, status_code=400)
        
        # Search the index
        results = search_index(query_embedding, k=10, threshold=similarity_threshold)
        
        return JSONResponse(content={"results": results}, status_code=200)
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Endpoint to get a video by ID
@app.get("/get_video/{video_id}")
async def get_video(video_id: str):
    try:
        # Find the video file
        video_files = [f for f in os.listdir("data/videos") if f.startswith(video_id)]
        if not video_files:
            return JSONResponse(content={"error": "Video not found"}, status_code=404)
        
        video_path = f"data/videos/{video_files[0]}"
        return FileResponse(video_path)
    except Exception as e:
        logger.error(f"Error in get_video: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Endpoint to get a frame by ID
@app.get("/get_frame/{video_id}/{frame_id}")
async def get_frame(video_id: str, frame_id: str):
    try:
        frame_path = f"data/frames/{video_id}/frame_{frame_id}.jpg"
        if not os.path.exists(frame_path):
            return JSONResponse(content={"error": "Frame not found"}, status_code=404)
        
        return FileResponse(frame_path)
    except Exception as e:
        logger.error(f"Error in get_frame: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Endpoint to get all frames for a video
@app.get("/video_frames/{video_id}")
async def get_video_frames(video_id: str):
    try:
        frames_dir = f"data/frames/{video_id}"
        if not os.path.exists(frames_dir):
            return JSONResponse(content={"error": "Video frames not found"}, status_code=404)
        
        # Get all frame metadata from the index
        video_frames = []
        for i, meta in enumerate(metadata):
            if meta.get("type") == "video_frame" and meta.get("video_id") == video_id:
                video_frames.append({
                    "frame_id": meta["frame_id"],
                    "timestamp": meta["timestamp"],
                    "timestamp_seconds": meta["timestamp_seconds"],
                    "path": f"/get_frame/{video_id}/{meta['frame_id']:04d}"
                })
        
        # Sort frames by timestamp
        video_frames.sort(key=lambda x: x["timestamp_seconds"])
        
        return JSONResponse(content={"data": video_frames}, status_code=200)
    except Exception as e:
        logger.error(f"Error in get_video_frames: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Endpoint to get analysis history
@app.get("/get_analysis_history")
async def get_analysis_history():
    try:
        # Get unique video IDs from metadata
        video_ids = set()
        for meta in metadata:
            if meta.get("type") == "video_frame" or meta.get("type") == "video_transcript":
                video_ids.add(meta.get("video_id"))
        
        # Get video details
        video_history = []
        for video_id in video_ids:
            # Find the video file
            video_files = [f for f in os.listdir("data/videos") if f.startswith(video_id)]
            if not video_files:
                continue
            
            # Get frame count
            frame_count = len([m for m in metadata if m.get("type") == "video_frame" and m.get("video_id") == video_id])
            
            # Get transcript segments count
            transcript_count = len([m for m in metadata if m.get("type") == "video_transcript" and m.get("video_id") == video_id])
            
            video_history.append({
                "video_id": video_id,
                "filename": video_files[0].replace(f"{video_id}_", ""),
                "frames_count": frame_count,
                "transcript_segments_count": transcript_count,
                "timestamp": datetime.fromtimestamp(os.path.getctime(f"data/videos/{video_files[0]}")).strftime("%Y-%m-%d %H:%M:%S")
            })
        
        return JSONResponse(content={"data": video_history}, status_code=200)
    except Exception as e:
        logger.error(f"Error in get_analysis_history: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)