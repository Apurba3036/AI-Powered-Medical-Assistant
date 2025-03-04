import os
import whisper
import gradio as gr
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, CSVLoader

# Load Whisper model
whisper_model = whisper.load_model("medium")

# Load FAISS for retrieval
pdf_loader = PyPDFLoader("Current Essentials of Medicine.pdf")
pdf_docs = pdf_loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
pdf_documents = text_splitter.split_documents(pdf_docs)

csv_folder = os.path.abspath("./csv_files")
csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]
csv_documents = []
for file in csv_files:
    try:
        csv_loader = CSVLoader(os.path.join(csv_folder, file), encoding="utf-8")
        csv_documents.extend(csv_loader.load())
    except Exception as e:
        print(f"Error loading {file}: {e}")

all_documents = pdf_documents + csv_documents
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
db = FAISS.from_documents(all_documents, embedding_model)
retriever = db.as_retriever(search_kwargs={"k": 3})

# Load LLM models

llm_gemma = Ollama(model="gemma:2b")  # For structured output

# Define prompts
retrieval_prompt = PromptTemplate.from_template("""
Answer the question based only on the provided medical context.
<context>
{context}
</context>
Question: {input}
""")

gemma_prompt = PromptTemplate(
    input_variables=["doctor_notes"],
    template="""
    You are an AI medical assistant. Convert the following doctor's notes into a structured format:

    **Doctor's Notes:** {doctor_notes}

    - **Patient Name**: (Leave blank if not mentioned)
    - **Diagnosis**: 
    - **Medications**: 
    - **Dosage Instructions**: 
    - **Additional Advice**: 
    - **Follow-up Instructions**: 
    """
)

retrieval_chain = create_retrieval_chain(retriever, LLMChain(llm=llm_gemma, prompt=retrieval_prompt))
llm_chain_gemma = LLMChain(llm=llm_gemma, prompt=gemma_prompt)

def transcribe_and_generate(audio):
    if not audio or not os.path.exists(audio):
        return "Error: No valid audio input detected."
    
    try:
        # Step 1: Convert speech to text
        audio_data = whisper.load_audio(audio)
        audio_data = whisper.pad_or_trim(audio_data)
        mel = whisper.log_mel_spectrogram(audio_data).to(whisper_model.device)
        options = whisper.DecodingOptions(fp16=False, language="en")
        result = whisper.decode(whisper_model, mel, options)
        transcribed_text = result.text
        
        # Step 2: Retrieve relevant medical information
        response = retrieval_chain.invoke({"input": transcribed_text})
        retrieved_answer = response.get("answer", "No relevant information found.")
        
        # Step 3: Process structured output with Gemma
        structured_output = llm_chain_gemma.run({"doctor_notes": retrieved_answer})
        
        return f"**Doctor's Transcription:** {transcribed_text}\n\n**Retrieved Medical Info:** {retrieved_answer}\n\n**Structured Output:**\n{structured_output}"
    except Exception as e:
        return f"Error processing audio: {str(e)}"

# Create Gradio interface
gr.Interface(
    fn=transcribe_and_generate,
    inputs=gr.Audio(sources=["microphone"], type="filepath"),
    outputs="textbox",
    title="AI Medical Assistant - Speech to Structured Medical Info",
    live=True,
).launch()
