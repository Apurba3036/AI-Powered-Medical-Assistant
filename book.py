import logging
import os
from typing import Dict, List
import base64
from io import BytesIO
from PIL import Image
import requests
import tempfile

from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.units import inch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Ollama LLM model
logger.info("Initializing Ollama LLM model...")
llm = Ollama(model="deepseek-r1:1.5b") 
logger.info("Ollama LLM initialized")

# MongoDB client setup
try:
    client = MongoClient("mongodb+srv://apurba:cfab9bS66QkZnsTs@cluster0.wznn11w.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    db = client.book_generator_db
    books_collection = db.books
    logger.info("MongoDB connection established")
except Exception as e:
    logger.error(f"MongoDB connection error: {str(e)}")
    logger.warning("Continuing without MongoDB connection")

# Replace the book_template in the FastAPI backend with this improved version:

# Define prompt template for book generation
book_template = PromptTemplate(
    input_variables=["topic", "pages"],
    template="""
    You are an expert book author and illustrator. Create a comprehensive book on the topic: {topic}.
    
    The book should be at least {pages} pages long with the following sections:
    
    1. Title Page (include a creative title)
    2. Introduction (what is this topic about and why is it important)
    3. Key Concepts (main ideas and principles)
    4. In-Depth Analysis (detailed exploration of the topic)
    5. Practical Applications (how this knowledge can be applied)
    6. Case Studies or Examples (illustrative examples)
    7. Future Perspectives (emerging trends or future directions)
    8. Conclusion (summary of key points)
    
    For each section, provide:
    1. A section heading
    2. Comprehensive content (at least 2-3 paragraphs per section)
    3. A brief description of an image that would enhance that section
    
    You MUST format your response in valid JSON format with this exact structure:
    
    {{
        "title": "The Book Title",
        "author": "AI Assistant",
        "sections": [
            {{
                "heading": "Section Title",
                "content": "Section content goes here with multiple paragraphs...",
                "image_prompt": "Description for generating an image related to this section"
            }},
            {{
                "heading": "Another Section Title",
                "content": "More section content...",
                "image_prompt": "Another image description"
            }}
        ]
    }}
    
    The JSON must be valid without any explanation text before or after it. Make sure all quotes are properly escaped and there are no trailing commas. Your entire response should be only valid JSON that can be parsed directly.
    """
)
# Create LLM chain for book generation
book_chain = LLMChain(llm=llm, prompt=book_template)

# Function to generate images using Hugging Face API
def generate_image(prompt: str):
    try:
        API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
        headers = {"Authorization": f"Bearer {os.environ.get('hf_dummy_','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')}"}
        
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
        
        if response.status_code != 200:
            logger.error(f"Error generating image: {response.text}")
            return None
            
        image_bytes = BytesIO(response.content)
        image = Image.open(image_bytes)
        
        # Convert to base64 for sending to frontend
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        logger.error(f"Image generation error: {str(e)}")
        return None

# Function to generate PDF
def generate_pdf(book_data: Dict):
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=30
        )
        
        # Add title
        story.append(Paragraph(book_data["title"], title_style))
        story.append(Paragraph(f"By {book_data['author']}", styles["Normal"]))
        story.append(Spacer(1, 2*inch))
        
        # Add sections
        for section in book_data["sections"]:
            story.append(Paragraph(section["heading"], styles["Heading1"]))
            story.append(Spacer(1, 0.2*inch))
            
            # Add content paragraphs
            content = section["content"].replace('\n\n', '<br/><br/>')
            story.append(Paragraph(content, styles["Normal"]))
            story.append(Spacer(1, 0.5*inch))
            
            # Add image if available
            if "image_data" in section and section["image_data"]:
                img_data = section["image_data"].split(',')[1]
                img_bytes = BytesIO(base64.b64decode(img_data))
                img = ReportLabImage(img_bytes, width=400, height=300)
                story.append(img)
                story.append(Spacer(1, 0.5*inch))
        
        doc.build(story)
        pdf_data = buffer.getvalue()
        buffer.close()
        
        return base64.b64encode(pdf_data).decode()
    except Exception as e:
        logger.error(f"PDF generation error: {str(e)}")
        return None

# Replace the generate_book function in the FastAPI backend with this more robust version:

@app.post("/generate_book")
async def generate_book(data: Dict = Body(...)):
    try:
        topic = data.get("topic", "")
        pages = data.get("pages", 5)
        
        if not topic:
            return JSONResponse(content={"error": "Topic is required"}, status_code=400)
            
        logger.info(f"Generating book content for topic: {topic}")
        
        # Generate book content using LLM
        raw_output = book_chain.run({"topic": topic, "pages": pages})
        
        # Handle JSON parsing with error recovery
        try:
            # Try direct parsing first
            import json
            book_data = json.loads(raw_output)
        except json.JSONDecodeError:
            logger.warning("Initial JSON parsing failed, attempting to fix output")
            
            # Try to extract JSON portion if wrapped in markdown or other text
            import re
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', raw_output)
            if json_match:
                try:
                    book_data = json.loads(json_match.group(1))
                    logger.info("Successfully extracted JSON from markdown code block")
                except json.JSONDecodeError:
                    pass
            
            # If still failing, try a more aggressive approach to repair the JSON
            if 'book_data' not in locals():
                logger.warning("Attempting to repair malformed JSON")
                # Find anything that looks like JSON
                json_pattern = r'({[\s\S]*})'
                potential_json = re.search(json_pattern, raw_output)
                
                if potential_json:
                    try:
                        # Try to clean and parse the JSON
                        potential_json_str = potential_json.group(1)
                        # Replace common issues
                        potential_json_str = re.sub(r',\s*}', '}', potential_json_str)  # Remove trailing commas
                        potential_json_str = re.sub(r',\s*]', ']', potential_json_str)  # Remove trailing commas in arrays
                        book_data = json.loads(potential_json_str)
                        logger.info("Successfully repaired JSON")
                    except json.JSONDecodeError:
                        pass
            
            # If all attempts failed, create a basic structure
            if 'book_data' not in locals():
                logger.error("Could not parse JSON, creating fallback structure")
                logger.error(f"Raw output: {raw_output}")
                
                # Create a basic book structure from the raw text
                book_data = {
                    "title": f"Book on {topic.title()}",
                    "author": "AI Assistant",
                    "sections": []
                }
                
                # Try to extract sections from raw text
                section_pattern = r'#+\s*(.*?)\n([\s\S]*?)(?=#+\s*|$)'
                sections = re.findall(section_pattern, raw_output)
                
                if sections:
                    for heading, content in sections:
                        # Find image descriptions
                        image_prompt = ""
                        img_match = re.search(r'Image:?\s*(.*?)(?=\n|$)', content)
                        if img_match:
                            image_prompt = img_match.group(1)
                            
                        book_data["sections"].append({
                            "heading": heading.strip(),
                            "content": content.strip(),
                            "image_prompt": image_prompt or f"Illustration of {heading.strip()}"
                        })
                else:
                    # If no sections found, create a default one
                    book_data["sections"].append({
                        "heading": "Introduction",
                        "content": raw_output[:1000],  # Use the first 1000 chars as content
                        "image_prompt": f"Illustration of {topic}"
                    })
        
        # Ensure the book data has the required structure
        if "title" not in book_data:
            book_data["title"] = f"Book on {topic.title()}"
        if "author" not in book_data:
            book_data["author"] = "AI Assistant"
        if "sections" not in book_data or not book_data["sections"]:
            book_data["sections"] = [{
                "heading": "Introduction",
                "content": f"This book explores the fascinating topic of {topic}.",
                "image_prompt": f"Illustration of {topic}"
            }]
        
        # Generate images for each section
        for section in book_data["sections"]:
            if "image_prompt" in section and section["image_prompt"]:
                logger.info(f"Generating image for prompt: {section['image_prompt']}")
                section["image_data"] = generate_image(section["image_prompt"])
        
        # Generate PDF
        pdf_base64 = generate_pdf(book_data)
        
        # Save to MongoDB
        if 'books_collection' in globals():
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            book_record = {
                "timestamp": timestamp,
                "topic": topic,
                "book_data": book_data,
                "pdf_base64": pdf_base64
            }
            books_collection.insert_one(book_record)
            logger.info("Book saved to MongoDB")
        
        return JSONResponse(content={
            "book_data": book_data,
            "pdf_base64": pdf_base64
        }, status_code=200)
    except Exception as e:
        logger.error(f"Error generating book: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
@app.get("/get_books")
async def get_books():
    try:
        if 'books_collection' not in globals():
            return JSONResponse(content={"data": []}, status_code=200)
            
        logger.info("Fetching books from MongoDB...")
        books = books_collection.find()
        book_list = [
            {
                "timestamp": book["timestamp"],
                "topic": book["topic"],
                "title": book["book_data"]["title"],
                "pdf_base64": book["pdf_base64"]
            }
            for book in books
        ]
        return JSONResponse(content={"data": book_list}, status_code=200)
    except Exception as e:
        logger.error(f"Error fetching books: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)