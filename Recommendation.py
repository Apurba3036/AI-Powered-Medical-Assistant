import os
import whisper
import gradio as gr
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, CSVLoader

# Load Whisper model
whisper_model = whisper.load_model("medium")

# # Load FAISS for retrieval
# pdf_loader = PyPDFLoader("Current Essentials of Medicine.pdf")
# pdf_docs = pdf_loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
# pdf_documents = text_splitter.split_documents(pdf_docs)

csv_folder = os.path.abspath("./csv_files")
csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]
csv_documents = []
for file in csv_files:
    try:
        csv_loader = CSVLoader(os.path.join(csv_folder, file), encoding="utf-8")
        csv_documents.extend(csv_loader.load())
    except Exception as e:
        print(f"Error loading {file}: {e}")

all_documents = csv_documents  
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.from_documents(all_documents, embedding_model)  # 🔹 Store all docs in FAISS
retriever = db.as_retriever(search_kwargs={"k": 2})

# Load LLM models
llm_gemma = Ollama(model="gemma:2b")  # For structured output

# Define prompts
retrieval_prompt = PromptTemplate.from_template("""
Use the following medical information to extract the symptoms and medicines from the input question and retrieve only relevant medicines from the context.
Give me the medicines names.
If no relevant match is found, return 'No relevant information found.'
Context: {context}
Question: {input}
""")

gemma_prompt = PromptTemplate(
    input_variables=["transcribed_text", "retrieved_answer"],
    template="""
    You are an AI medical assistant. Analyze the following doctor's transcription and the retrieved medical information both. 
    Convert them into a structured format:
    

    **Doctor's text:** {transcribed_text}

    **Retrieved Medical Info:** {retrieved_answer}
    
     Format the prescription like this:
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
        # print("Retrieval Response:", response)  # 🔹 Debugging
        retrieved_answer = response.get("answer", "No relevant information found.")  # 🔹 Adjust key
        print(retrieved_answer)

        # Step 3: Process structured output with Gemma
        response = llm_chain_gemma.invoke({"transcribed_text": transcribed_text, "retrieved_answer": retrieved_answer})
        print("Gemma Response:", response)  # 🔹 Debugging
        structured_output = response.get("text", "Error: No structured output generated.")  # 🔹 Adjust key
        # print(structured_output)

        return f"**Doctor's Transcription:** {transcribed_text}\n\n**Structured Output:**\n{structured_output}"
    except Exception as e:
        return f"Error processing audio: {str(e)}"

# Create Gradio interface
gr.Interface(
    fn=transcribe_and_generate,
    inputs=gr.Audio(sources=["microphone"], type="filepath"),
    outputs="textbox",
    title="AI Medical Assistant - Speech to Prescription",
    live=True,
).launch()
