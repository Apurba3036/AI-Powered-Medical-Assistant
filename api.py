import tempfile
import os
import logging
import gridfs
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
import whisper
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from datetime import datetime

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

# Initialize Whisper model
logger.info("Loading Whisper model...")
whisper_model = whisper.load_model("medium")
logger.info("Whisper model loaded successfully")

# Initialize Ollama LLM model
logger.info("Initializing Ollama LLM model...")
llm = Ollama(model="gemma:2b")
logger.info("Ollama LLM initialized")

# Define prompt for prescription formatting
prompt_template = PromptTemplate(
    input_variables=["doctor_notes"],
    template=""" 
    You are an AI medical assistant. The following is a doctor's dictated prescription.
    Convert it into a structured and professional format.

    Doctor's Notes: {doctor_notes}

    Format the prescription like this:
    - **Patient Name**:
    - **Diagnosis**: 
    - **Medications**: 
    - **Dosage Instructions**: 
    - **Additional Advice**: 
    - **Follow-up Instructions**: 
    """
)

llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# MongoDB client setup
try:
    client = MongoClient("mongodb+srv://apurba:cfab9bS66QkZnsTs@cluster0.wznn11w.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    db = client.medical_db
    prescriptions_collection = db.prescriptions
    # Initialize GridFS for storing audio files
    fs = gridfs.GridFS(db)
    logger.info("MongoDB connection established")
except Exception as e:
    logger.error(f"MongoDB connection error: {str(e)}")
    raise

def transcribe_and_generate(audio_path, audio_file_id=None):
    try:
        logger.info(f"Starting transcription process for file: {audio_path}")
        
        # Ensure file exists
        if not os.path.exists(audio_path):
            logger.error(f"Audio file does not exist: {audio_path}")
            return {"error": "Audio file not found"}
            
        # Check file size
        file_size = os.path.getsize(audio_path)
        logger.info(f"Audio file size: {file_size} bytes")
        
        if file_size == 0:
            logger.error("Audio file is empty")
            return {"error": "Audio file is empty"}
        
        # Load and transcribe audio using Whisper's load_audio function
        try:
            audio_data = whisper.load_audio(audio_path)
            logger.info(f"Audio loaded: {len(audio_data)} samples")
            
            audio_data = whisper.pad_or_trim(audio_data)
            mel = whisper.log_mel_spectrogram(audio_data).to(whisper_model.device)
            
            options = whisper.DecodingOptions(fp16=False, language="en")
            result = whisper.decode(whisper_model, mel, options)
            transcribed_text = result.text
            logger.info(f"Transcribed text: {transcribed_text}")
            
            if len(transcribed_text.strip()) < 5:
                logger.warning("Transcription produced very little text, possible audio quality issue")
            
            # Generate structured prescription
            structured_prescription = llm_chain.run({"doctor_notes": transcribed_text})
            logger.info("Structured prescription generated")
            
            # Save to MongoDB (including audio file id if available)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            prescription_data = {
                "timestamp": timestamp,
                "doctor_notes": transcribed_text,
                "structured_prescription": structured_prescription,
                "audio_file_id": str(audio_file_id) if audio_file_id is not None else None
            }
            prescriptions_collection.insert_one(prescription_data)
            logger.info("Data saved to MongoDB")
            
            return {
                "doctor_notes": transcribed_text,
                "structured_prescription": structured_prescription,
                "audio_file_id": str(audio_file_id) if audio_file_id is not None else None
            }
        except Exception as e:
            logger.error(f"Error in Whisper processing: {str(e)}")
            return {"error": f"Failed to process audio: {str(e)}"}
    except Exception as e:
        logger.error(f"Error during transcription or prescription generation: {str(e)}")
        return {"error": str(e)}

@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    try:
        logger.info(f"Receiving audio file: {file.filename}, content_type: {file.content_type}")
        
        # Create a temporary file with the correct extension based on content type
        extension = ".webm"
        if "ogg" in file.content_type:
            extension = ".ogg"
        elif "mp4" in file.content_type or "mp3" in file.content_type:
            extension = ".mp4"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
            # Read the file in chunks to handle large files
            chunk_size = 1024 * 1024  # 1MB chunks
            content = await file.read(chunk_size)
            while content:
                temp_file.write(content)
                content = await file.read(chunk_size)
            temp_file_path = temp_file.name
        
        logger.info(f"Audio saved to temporary file: {temp_file_path}")
        
        # Save the audio file in GridFS
        with open(temp_file_path, "rb") as f:
            file_data = f.read()
        audio_file_id = fs.put(file_data, filename=file.filename)
        logger.info(f"Audio file saved in GridFS with id: {audio_file_id}")
        
        # Process the audio file (transcription and prescription generation)
        response = transcribe_and_generate(temp_file_path, audio_file_id)
        
        # Clean up the temporary file
        try:
            os.unlink(temp_file_path)
            logger.info(f"Temporary file deleted: {temp_file_path}")
        except Exception as e:
            logger.warning(f"Could not delete temporary file: {str(e)}")
        
        if "error" in response:
            logger.error(f"Error response: {response['error']}")
            return JSONResponse(content={"error": response["error"]}, status_code=500)
        
        return JSONResponse(content=response, status_code=200)
    except Exception as e:
        logger.error(f"Error in upload_audio: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/get_prescriptions")
async def get_prescriptions():
    try:
        logger.info("Fetching prescriptions from MongoDB...")
        prescriptions = prescriptions_collection.find()
        prescription_list = [
            {
                "timestamp": pres["timestamp"], 
                "doctor_notes": pres["doctor_notes"], 
                "structured_prescription": pres["structured_prescription"],
                "audio_file_id": pres.get("audio_file_id")
            } 
            for pres in prescriptions
        ]
        return JSONResponse(content={"data": prescription_list}, status_code=200)
    except Exception as e:
        logger.error(f"Error fetching prescriptions: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
