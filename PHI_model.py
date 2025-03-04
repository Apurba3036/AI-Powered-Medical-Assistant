import torch
import os
import gradio as gr
from transformers import AutoModelForCausalLM, AutoProcessor

# Load Phi-4 Multimodal Model
model_name = "microsoft/Phi-4-multimodal-instruct"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
)

# Function to Transcribe Audio Using Phi-4
def transcribe_audio_with_phi4(audio_path):
    with open(audio_path, "rb") as audio_file:
        audio_data = audio_file.read()

    inputs = processor(audio=audio_data, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    output_tokens = model.generate(**inputs, max_length=1024)
    transcription = processor.batch_decode(output_tokens, skip_special_tokens=True)[0]

    return transcription

# Function to Generate Structured Prescription
def generate_text(prompt):
    inputs = processor(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(**inputs, max_length=500)
    return processor.batch_decode(outputs, skip_special_tokens=True)[0]

# Main Function: Process Audio and Generate Prescription
def transcribe_and_generate(audio):
    print(f"Received audio file: {audio}") 
    if not audio or not os.path.exists(audio):
        return "Error: No valid audio input detected."
    
    try:
        # 🎤 Step 1: Transcribe Speech using Phi-4
        transcribed_text = transcribe_audio_with_phi4(audio)

        # ✨ Step 2: Generate Structured Prescription
        prompt = f"""
        You are an AI medical assistant. The following is a doctor's dictated prescription.
        Convert it into a structured and professional format.

        Doctor's Notes: {transcribed_text}

        Format the prescription like this:
        - **Patient Name**: (Leave blank if not mentioned)
        - **Diagnosis**: 
        - **Medications**: 
        - **Dosage Instructions**: 
        - **Additional Advice**: 
        - **Follow-up Instructions**: 
        """
        structured_prescription = generate_text(prompt)

        return f"**Doctor's Transcription:** {transcribed_text}\n\n**Formatted Prescription:**\n{structured_prescription}"
    
    except Exception as e:
        return f"Error processing audio: {str(e)}"

# Gradio Interface
gr.Interface(
    fn=transcribe_and_generate,
    inputs=gr.Audio(sources=["microphone"], type="filepath"),
    outputs="textbox",
    title="AI Medical Assistant - Doctor's Speech to Prescription",
    live=True,
).launch()
