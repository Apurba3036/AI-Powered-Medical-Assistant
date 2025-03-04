import whisper
import gradio as gr
import os
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


whisper_model = whisper.load_model("medium")


llm = Ollama(model="gemma:2b")


prompt_template = PromptTemplate(
    input_variables=["doctor_notes"],
    template="""
    You are an AI medical assistant. The following is a doctor's dictated prescription.
    Convert it into a structured and professional format.

    Doctor's Notes: {doctor_notes}

    Format the prescription like this:
    - **Patient Name**: (Leave blank if not mentioned)
    - **Diagnosis**: 
    - **Medications**: 
    - **Dosage Instructions**: 
    - **Additional Advice**: 
    - **Follow-up Instructions**: 
    """
)


llm_chain = LLMChain(llm=llm, prompt=prompt_template)


def transcribe_and_generate(audio):
    print(f"Received audio file: {audio}") 
    if not audio or not os.path.exists(audio):
        return "Error: No valid audio input detected."
    
    try:
        
        audio_data = whisper.load_audio(audio)
        audio_data = whisper.pad_or_trim(audio_data)
        mel = whisper.log_mel_spectrogram(audio_data).to(whisper_model.device)

    
        options = whisper.DecodingOptions(fp16=False, language="en")
        result = whisper.decode(whisper_model, mel, options)
        transcribed_text = result.text

        
        structured_prescription = llm_chain.run({"doctor_notes": transcribed_text})

        return f"**Doctor's Transcription:** {transcribed_text}\n\n**Formatted Prescription:**\n{structured_prescription}"
    
    except Exception as e:
        return f"Error processing audio: {str(e)}"

# Create Gradio interface
gr.Interface(
    fn=transcribe_and_generate,
    inputs=gr.Audio(sources=["microphone"], type="filepath"),
    outputs="textbox",
    title="AI Medical Assistant - Doctor's Speech to Prescription",
    live=True,  # Enables real-time processing
   
).launch()

