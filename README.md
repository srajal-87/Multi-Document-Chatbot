# 📚 Smart PDF Chat: Multi-Document Chatbot

## 📌 Overview

Smart PDF Chat is an intelligent conversational assistant that enables users to interact with multiple PDF documents through natural language questions.

This application allows users to upload PDFs and quickly extract insights without having to manually read through large documents. Perfect for researchers, students, professionals, and anyone needing to efficiently analyze document collections.

## 🚀 Features

- **Multiple PDF Support**: Upload and process several documents simultaneously
- **Context-Aware Q&A**: Ask questions about document content and receive accurate answers
- **Source Citation**: Responses include document name and page number references
- **Smart Text Processing**: Advanced chunking and embedding for better understanding
- **User-Friendly Interface**: Clean Streamlit interface with chat-like experience
- **Markdown Formatting**: Responses are formatted for improved readability

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Language Models**: Groq API (deepseek-r1-distill-llama-70b)
- **Embeddings**: Hugging Face (sentence-transformers/all-MiniLM-L6-v2)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Document Processing**: PyPDF2, LangChain
- **Language**: Python 3.8+

## 📁 Project Structure

```
smart-pdf-chat/
├── app.py                # Main application file
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (API keys)
├── README.md             # Project documentation
├── venv/                 # Virtual environment (created during setup)
└── .gitignore            # Git ignore file
```

## 🧠 How It Works

1. **Document Processing**:
   - PDFs are uploaded and text is extracted with page tracking
   - Text is split into meaningful chunks with source information preserved
   - Chunks are embedded and stored in a FAISS vector database

2. **Query Processing**:
   - User questions are embedded and compared to stored document chunks
   - Most relevant chunks are retrieved
   - LLM generates responses using the retrieved context
   - Response is formatted with source citations

3. **Conversation Management**:
   - Chat history is maintained for context
   - Follow-up questions work with previous context

## 🧪 Setup & Installation

### Prerequisites
- Python 3.8+
- Groq API Key
- Hugging Face Token (optional)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/smart-pdf-chat.git
cd smart-pdf-chat
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure API Keys
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token  # Optional
```

### Step 5: Run the Application
```bash
streamlit run app.py
```

## 📊 Usage Flow

1. **Upload Documents**: Use the sidebar to upload one or more PDF files
2. **Process Documents**: Click "Process Documents" button
3. **Ask Questions**: Type your queries about the document content
4. **Review Answers**: Get context-aware responses with source citations

## 📈 Future Improvements

- **Multi-format Support**: Add support for DOCX, TXT, and other file formats
- **Custom Embedding Options**: Allow users to select different embedding models
- **Enhanced Visualization**: Add document summarization and content visualization
- **Multilingual Support**: Process and answer questions in multiple languages
- **Export Functionality**: Save conversations and insights
- **Fine-tuning Options**: Allow users to customize response length and detail level

## 🧑‍💻 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
