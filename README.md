# 📚 Smart PDF Chat: Multi-Document Chatbot

## Overview
Smart PDF Chat is an intelligent Streamlit application that allows users to upload multiple PDF documents and interact with their content through an AI-powered conversational interface.

## Features
- 🔤 Multi-PDF document processing
- 💬 Context-aware question answering
- 🔍 Advanced text extraction and chunking
- 📊 Detailed document statistics

## Prerequisites
- Python 3.8+
- Groq API Key
- Hugging Face Token (optional)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/smart-pdf-chat.git
cd smart-pdf-chat
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token  # Optional
```

## Running the Application
```bash
streamlit run app.py
```

## Usage
1. Upload PDF documents
2. Click "Process Documents"
3. Start asking questions about the uploaded documents

## Configuration Options
- Adjust `chunk_size` and `chunk_overlap` for text splitting
- Modify LLM temperature in `initialize_chat_model`
- Change embedding model in `create_vector_store`

## Technologies
- Streamlit
- LangChain
- Groq API
- Hugging Face Embeddings
- FAISS Vector Store