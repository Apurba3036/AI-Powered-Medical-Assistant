import tempfile
import os
import logging
import gridfs
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
import whisper
from PIL import Image
from io import BytesIO
import torch
import mimetypes
from transformers import AutoProcessor, AutoModelForCausalLM
from datetime import datetime
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

# Initialize models
logger.info("Loading models...")

# Initialize Whisper model for audio transcription
whisper_model = whisper.load_model("medium")
logger.info("Whisper model loaded successfully")

# Initialize Ollama LLM model
llm = Ollama(model="gemma:2b")
logger.info("Ollama LLM initialized")

# Initialize a multimodal model from Hugging Face
processor = AutoProcessor.from_pretrained("microsoft/git-base")
multimodal_model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")
logger.info("Multimodal transformer loaded successfully")

# MongoDB client setup
try:
    client = MongoClient("mongodb+srv://apurba:cfab9bS66QkZnsTs@cluster0.wznn11w.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    db = client.multimodal_db
    analysis_collection = db.analyses
    # Initialize GridFS for storing files
    fs = gridfs.GridFS(db)
    logger.info("MongoDB connection established")
except Exception as e:
    logger.error(f"MongoDB connection error: {str(e)}")
    raise

# Create LangChain tools for different input types
@tool
def analyze_text(text: str) -> str:
    """Analyzes text input and returns insights."""
    try:
        logger.info("Analyzing text with multimodal model")
        inputs = processor(text=text, return_tensors="pt")
        with torch.no_grad():
            outputs = multimodal_model.generate(
                inputs["input_ids"], 
                max_length=100,
            )
        generated_text = processor.decode(outputs[0], skip_special_tokens=True)
        
        # Use LLM to structure the analysis
        template = """
        Analyze the following text and provide a structured analysis:
        
        TEXT: {text}
        
        INITIAL ANALYSIS: {initial_analysis}
        
        Provide a comprehensive analysis with the following:
        - Main topic/theme
        - Key points
        - Sentiment
        - Summary
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["text", "initial_analysis"]
        )
        
        chain = prompt | llm | StrOutputParser()
        
        result = chain.invoke({"text": text, "initial_analysis": generated_text})
        return result
    except Exception as e:
        logger.error(f"Error in text analysis tool: {str(e)}")
        return f"Error analyzing text: {str(e)}"

@tool
def analyze_image(image_path: str) -> str:
    """Analyzes image content and returns insights."""
    try:
        logger.info(f"Analyzing image: {image_path}")
        
        # Load image
        image = Image.open(image_path)
        
        # Process image with multimodal model
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = multimodal_model.generate(
                pixel_values=inputs.pixel_values,
                max_length=100,
            )
        generated_description = processor.decode(outputs[0], skip_special_tokens=True)
        
        # Use LLM to structure the analysis
        template = """
        Analyze the following image description and provide a structured analysis:
        
        IMAGE DESCRIPTION: {image_description}
        
        Provide a comprehensive analysis with the following:
        - Main content identification
        - Visual elements breakdown
        - Context interpretation
        - Possible applications
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["image_description"]
        )
        
        chain = prompt | llm | StrOutputParser()
        
        result = chain.invoke({"image_description": generated_description})
        return result
    except Exception as e:
        logger.error(f"Error in image analysis tool: {str(e)}")
        return f"Error analyzing image: {str(e)}"

@tool
def analyze_audio(audio_path: str) -> dict:
    """Transcribes audio and analyzes the content."""
    try:
        logger.info(f"Transcribing audio: {audio_path}")
        
        # Load and transcribe audio using Whisper
        audio_data = whisper.load_audio(audio_path)
        audio_data = whisper.pad_or_trim(audio_data)
        mel = whisper.log_mel_spectrogram(audio_data).to(whisper_model.device)
        
        options = whisper.DecodingOptions(fp16=False, language="en")
        result = whisper.decode(whisper_model, mel, options)
        transcribed_text = result.text
        
        logger.info(f"Transcribed text: {transcribed_text}")
        
        # Analyze the transcribed text
        template = """
        Analyze the following audio transcript and provide a structured analysis:
        
        TRANSCRIPT: {transcript}
        
        Provide a comprehensive analysis with the following:
        - Main topic/subject
        - Key points
        - Speaker tone and emotion
        - Summary
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["transcript"]
        )
        
        chain = prompt | llm | StrOutputParser()
        
        result = chain.invoke({"transcript": transcribed_text})
        return {"transcript": transcribed_text, "analysis": result}
    except Exception as e:
        logger.error(f"Error in audio analysis tool: {str(e)}")
        return {"error": f"Error analyzing audio: {str(e)}"}

# Helper function to determine file type
def determine_file_type(file_path, content_type):
    """Determine if the file is an image, audio, or text based on MIME type."""
    if content_type.startswith('image/'):
        return 'image'
    elif content_type.startswith('audio/') or content_type in ['video/webm', 'video/mp4']:
        return 'audio'
    else:
        return 'text'

# Define a single unified API endpoint
@app.post("/analyze")
async def analyze_content(file: UploadFile = File(None), text: str = Form(None)):
    try:
        # Determine the input type
        if file:
            # Get the content type
            content_type = file.content_type
            file_type = determine_file_type(file.filename, content_type)
            logger.info(f"Received {file_type} file: {file.filename}, content_type: {content_type}")
            
            # Create a temporary file
            extension = os.path.splitext(file.filename)[1]
            if not extension:
                extension = mimetypes.guess_extension(content_type) or '.tmp'
                
            with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
                # Read the file in chunks to handle large files
                chunk_size = 1024 * 1024  # 1MB chunks
                content = await file.read(chunk_size)
                while content:
                    temp_file.write(content)
                    content = await file.read(chunk_size)
                temp_file_path = temp_file.name
            
            logger.info(f"File saved to temporary file: {temp_file_path}")
            
            # Save the file in GridFS
            with open(temp_file_path, "rb") as f:
                file_data = f.read()
            file_id = fs.put(file_data, filename=file.filename, content_type=content_type)
            logger.info(f"File saved in GridFS with id: {file_id}")
            
            # Process based on file type
            if file_type == 'image':
                analysis_result = analyze_image(temp_file_path)
                transcript = None
            elif file_type == 'audio':
                audio_result = analyze_audio(temp_file_path)
                analysis_result = audio_result.get('analysis', '')
                transcript = audio_result.get('transcript', '')
            else:
                # If it's a text file, read it and process as text
                with open(temp_file_path, 'r', errors='ignore') as f:
                    file_text = f.read()
                analysis_result = analyze_text(file_text)
                transcript = None
            
            # Clean up the temporary file
            try:
                os.unlink(temp_file_path)
                logger.info(f"Temporary file deleted: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Could not delete temporary file: {str(e)}")
            
            # Save to MongoDB
            document = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "content_type": file_type,
                "file_id": str(file_id),
                "filename": file.filename,
                "analysis": analysis_result
            }
            
            if transcript:
                document["transcript"] = transcript
            
            result = analysis_collection.insert_one(document)
            logger.info(f"Analysis saved to MongoDB with ID: {result.inserted_id}")
            
            response_content = {
                "content_type": file_type,
                "analysis": analysis_result,
                "document_id": str(result.inserted_id)
            }
            
            if transcript:
                response_content["transcript"] = transcript
                
            return JSONResponse(content=response_content, status_code=200)
            
        elif text:
            logger.info("Received text for analysis")
            analysis_result = analyze_text(text)
            
            # Save to MongoDB
            document = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "content_type": "text",
                "original_content": text,
                "analysis": analysis_result
            }
            
            result = analysis_collection.insert_one(document)
            logger.info(f"Analysis saved to MongoDB with ID: {result.inserted_id}")
            
            return JSONResponse(content={
                "content_type": "text",
                "analysis": analysis_result,
                "document_id": str(result.inserted_id)
            }, status_code=200)
        else:
            return JSONResponse(content={"error": "No input provided"}, status_code=400)
    except Exception as e:
        logger.error(f"Error in unified analysis: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/get_analyses")
async def get_analyses(content_type: str = None):
    try:
        logger.info(f"Fetching analyses from MongoDB, filter by content_type: {content_type}")
        
        # Build the query based on whether a content_type filter was provided
        query = {}
        if content_type and content_type in ["text", "image", "audio"]:
            query["content_type"] = content_type
        
        analyses = analysis_collection.find(query).sort("timestamp", -1)
        
        # Format the results
        analysis_list = []
        for analysis in analyses:
            item = {
                "document_id": str(analysis["_id"]),
                "timestamp": analysis["timestamp"],
                "content_type": analysis["content_type"],
                "analysis": analysis["analysis"]
            }
            
            # Add content-type specific fields
            if analysis["content_type"] == "text":
                item["original_content"] = analysis.get("original_content", "")
            elif analysis["content_type"] in ["image", "audio"]:
                item["filename"] = analysis.get("filename", "")
                item["file_id"] = analysis.get("file_id", "")
            
            # Add transcript for audio
            if analysis["content_type"] == "audio":
                item["transcript"] = analysis.get("transcript", "")
                
            analysis_list.append(item)
        
        return JSONResponse(content={"data": analysis_list}, status_code=200)
    except Exception as e:
        logger.error(f"Error fetching analyses: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)