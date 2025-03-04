import os
import chainlit as cl
from langchain.chains import create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# 🔹 Step 1: Load and Process PDF
pdf_loader = PyPDFLoader("Current Essentials of Medicine.pdf")
pdf_docs = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
pdf_documents = text_splitter.split_documents(pdf_docs)

# 🔹 Step 2: Load and Process CSV Files
csv_folder = os.path.abspath("./csv_files")  # Ensure correct path
csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

csv_documents = []
for file in csv_files:
    file_path = os.path.join(csv_folder, file)
    print(f"Loading file: {file_path}")

    try:
        csv_loader = CSVLoader(file_path, encoding="utf-8")  # Use UTF-8 encoding
        docs = csv_loader.load()
        csv_documents.extend(docs)
    except Exception as e:
        print(f"Error loading {file}: {e}")

print(f"Total documents loaded: {len(csv_documents)}")

# 🔹 Step 3: Combine PDF and CSV Documents
all_documents = pdf_documents + csv_documents

# 🔹 Step 4: Create FAISS Vector Store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2", model_kwargs={"device": "cpu"})
db = FAISS.from_documents(all_documents, embedding_model)

# 🔹 Step 5: Create Retriever
retriever = db.as_retriever(search_kwargs={"k": 3})

# 🔹 Step 6: Load Ollama LLM
llm = Ollama(model="deepseek-r1:1.5b")  # Ensure this model exists

# 🔹 Step 7: Define Chat Prompt Template
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
I will tip you $1000 if the user finds the answer helpful. 

<context>
{context}
</context>

Question: {input}
""")

# 🔹 Step 8: Create Document Chain
document_chain = create_stuff_documents_chain(llm, prompt)

# 🔹 Step 9: Create Retrieval Chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# 🔹 Step 10: Initialize Memory
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set(
        "memory",
        ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="answer"
        )
    )

# Handle incoming messages
@cl.on_message
async def on_message(message):
    try:
        # Pass user input to retrieval chain
        response = retrieval_chain.invoke({"input": message.content})

        print("LLM Response:", response)  # Debugging

        # Extract answer
        answer = response.get("answer", "I'm not sure how to respond.")

    except Exception as e:
        print("Error:", e)
        answer = "An error occurred while processing your request."

    # Send the response back to the user
    await cl.Message(content=answer).send()

if __name__ == "__main__":
    cl.run()