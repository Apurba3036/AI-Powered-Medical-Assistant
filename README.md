# 🏥 Medical AI Assistant

## 📌 Project Overview
This project is a **Medical AI Assistant** that processes doctor's transcriptions, retrieves relevant medical information from structured datasets, and provides structured medical insights. It leverages **LangChain** for natural language processing and **FAISS** for efficient similarity search.

## 📂 Directory Structure
```
📦 Medical AI Assistant
┣ 📂 data
┃ ┣ 📄 brands_details_herbal.csv
┃ ┣ 📄 brands_details_allopathic.csv
┣ 📂 models
┃ ┣ 📄 embedding_model.py
┃ ┣ 📄 retrieval_model.py
┣ 📂 api
┃ ┣ 📄 app.py
┃ ┣ 📄 routes.py
┣ 📂 utils
┃ ┣ 📄 preprocessor.py
┃ ┣ 📄 vectorizer.py
┣ 📄 README.md
┣ 📄 requirements.txt
┣ 📄 config.py
```

## 🔍 Key Features
✅ **Doctor's Transcription Processing**: Extracts patient symptoms from text.
✅ **Medical Information Retrieval**: Finds relevant medicines and details from the database.
✅ **Vector-Based Search**: Uses **FAISS** for fast and efficient similarity search.
✅ **Structured Output**: Generates organized medical reports for easy interpretation.
✅ **Multi-CSV Support**: Handles both **herbal** and **allopathic** medicine datasets.

## 🏗️ Tech Stack
- **Backend**: FastAPI
- **Embedding Model**: `sentence-transformers/all-mpnet-base-v2`
- **Database**: MongoDB (for patient records) & FAISS (for vector embeddings)
- **Data Processing**: Pandas, NumPy
- **Deployment**: Docker (optional)

## 📖 Model & Database Structure
### **1️⃣ Embedding Model**
- Utilizes **`sentence-transformers/all-mpnet-base-v2`** to generate vector embeddings of medical data.
- Stores embeddings in **FAISS** for quick retrieval.

### **2️⃣ Database Schema** (MongoDB)
```json
{
  "_id": "ObjectId",
  "patient_name": "string",
  "symptoms": "string",
  "retrieved_info": {
    "medicine": "string",
    "dosage": "string",
    "side_effects": "string",
    "link": "string"
  },
  "timestamp": "datetime"
}
```

## 🔄 Workflow
1️⃣ **Doctor provides a transcription** (e.g., "My patient Anika has a fever.")
2️⃣ **NLP extracts symptoms** and processes the text.
3️⃣ **FAISS searches for relevant medicines** based on embeddings.
4️⃣ **Structured output is generated** with proper formatting.
5️⃣ **Doctor receives a summarized report** with medicine details.

## 🚀 Future Improvements
🔹 **Improve retrieval accuracy** with advanced embeddings.
🔹 **Integrate API for real-time responses**.
🔹 **Expand dataset with more medical records**.

## 🛠️ Setup & Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/Medical-AI-Assistant.git
cd Medical-AI-Assistant

# Install dependencies
pip install -r requirements.txt

# Run the API
python api/app.py
```

## 🤝 Contributing
Bacbon Limited



---
💡 **Let's revolutionize medical AI together!** 🚀

