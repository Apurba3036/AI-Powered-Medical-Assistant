import chainlit as cl
from langchain.chains import create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load and process PDF
loader = PyPDFLoader("Current Essentials of Medicine.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
documents = text_splitter.split_documents(docs)

# Create FAISS vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
db = FAISS.from_documents(documents, embedding_model)

# Create retriever
retriever = db.as_retriever(search_kwargs={"k": 3})

# Load Ollama Llama2 Model
llm = Ollama(model="gemma:2b")

# Define Chat Prompt Template
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
I will tip you $1000 if the user finds the answer helpful. 
<context>
{context}
</context>
Question: {input}
""")

# Create Stuff Document Chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Create Retrieval Chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Reset memory and create a new session memory for each new chat
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
