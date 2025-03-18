# import os
# import json
# import logging
# from io import BytesIO
# from typing import Dict
# from fastapi import FastAPI, Body
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pymongo import MongoClient
# from PIL import Image
# import requests
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from datetime import datetime
# from diffusers import StableDiffusionXLPipeline

# # Logging setup
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)

# # FastAPI app initialization
# app = FastAPI()

# # CORS setup
# origins = ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000"]
# app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# # # Hugging Face authentication token
# # HF_TOKEN = os.getenv("HF_TOKEN", "")

# # Load Hugging Face LLM model
# MODEL_NAME = "Qwen/QwQ-32B"  # Change this model if needed
# logger.info(f"Loading Hugging Face model: {MODEL_NAME}...")

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
# text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# logger.info("Hugging Face LLM loaded successfully.")

# # Load Stable Diffusion XL model for image generation
# logger.info("Loading Stable Diffusion XL model for image generation...")
# image_pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", use_auth_token=HF_TOKEN)
# logger.info("Stable Diffusion model loaded successfully.")

# # MongoDB setup
# try:
#     client = MongoClient("mongodb+srv://apurba:cfab9bS66QkZnsTs@cluster0.wznn11w.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
#     db = client.book_generator_db
#     books_collection = db.books
#     logger.info("Connected to MongoDB")
# except Exception as e:
#     logger.error(f"MongoDB connection error: {str(e)}")

# # Prompt template for book generation
# book_template = PromptTemplate(
#     input_variables=["topic", "pages"],
#     template="""
#     You are an expert book author. Write a book on the topic: {topic}.
    
#     The book should be {pages} pages long and must contain:

#     - Title Page
#     - Introduction
#     - Key Concepts
#     - In-Depth Analysis
#     - Practical Applications
#     - Case Studies
#     - Future Trends
#     - Conclusion

#     Format your response as JSON with:
    
#     {{
#         "title": "The Book Title",
#         "author": "AI Assistant",
#         "sections": [
#             {{
#                 "heading": "Section Title",
#                 "content": "Detailed section content...",
#                 "image_prompt": "A description of an image related to this section"
#             }}
#         ]
#     }}
#     """
# )

# # Hugging Face LLM wrapper for LangChain
# class HuggingFaceLLM:
#     def __init__(self, text_generator):
#         self.text_generator = text_generator

#     def __call__(self, prompt):
#         response = self.text_generator(prompt, max_length=2048, temperature=0.7, num_return_sequences=1)
#         return response[0]["generated_text"]

# hf_llm = HuggingFaceLLM(text_generator)
# book_chain = LLMChain(llm=hf_llm, prompt=book_template)

# # Function to generate an image using Stable Diffusion
# def generate_image(image_prompt: str):
#     try:
#         logger.info(f"Generating image for: {image_prompt}")
#         image = image_pipe(image_prompt).images[0]
        
#         # Convert image to base64
#         buffered = BytesIO()
#         image.save(buffered, format="PNG")
#         img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

#         return img_str
#     except Exception as e:
#         logger.error(f"Error generating image: {str(e)}")
#         return None

# # Route to generate book content with images
# @app.post("/generate_book")
# async def generate_book(data: Dict = Body(...)):
#     try:
#         topic = data.get("topic", "")
#         pages = data.get("pages", 5)

#         if not topic:
#             return JSONResponse(content={"error": "Topic is required"}, status_code=400)

#         logger.info(f"Generating book for topic: {topic}")

#         # Generate book content
#         raw_output = book_chain.run({"topic": topic, "pages": pages})

#         try:
#             book_data = json.loads(raw_output)
#         except json.JSONDecodeError:
#             logger.warning("Invalid JSON format from AI. Returning raw text.")
#             return JSONResponse(content={"error": "Invalid JSON format", "data": raw_output}, status_code=500)

#         # Generate images for each section
#         for section in book_data.get("sections", []):
#             image_prompt = section.get("image_prompt", "")
#             if image_prompt:
#                 image_data = generate_image(image_prompt)
#                 section["image"] = image_data if image_data else "Image generation failed"

#         # Save book to MongoDB
#         book_data["created_at"] = datetime.utcnow()
#         books_collection.insert_one(book_data)

#         return JSONResponse(content=book_data)

#     except Exception as e:
#         logger.error(f"Error generating book: {str(e)}")
#         return JSONResponse(content={"error": str(e)}, status_code=500)

# # Run FastAPI server
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
