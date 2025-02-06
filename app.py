import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

class PDFChatbot:
    def __init__(self):
        self._initialize_session_state()
        load_dotenv()
        self._load_api_keys()

    def _load_api_keys(self):
        """Load API keys from environment variables."""
        os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
        os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

    def _initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'conversation' not in st.session_state:
            st.session_state.conversation = None
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'processed_docs' not in st.session_state:
            st.session_state.processed_docs = {}

    def extract_pdf_text(self, pdf_docs):
        """Extract text from uploaded PDFs with page tracking."""
        text = ""
        docs_info = {}
        
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            docs_info[pdf.name] = len(pdf_reader.pages)
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text += f"\nDocument: {pdf.name} | Page: {page_num}\n"
                text += page.extract_text()
                text += "\n---\n"
                
        return text, docs_info

    def create_text_chunks(self, text):
        """Split text into manageable chunks with document source preservation."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n---\n", "\n", " ", ""]
        )
        return text_splitter.split_text(text)

    def create_vector_store(self, text_chunks):
        """Create vector store from text chunks."""
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        return FAISS.from_texts(text_chunks, embedding=embeddings)

    def initialize_chat_model(self, vector_store):
        """Initialize conversational retrieval chain with improved prompting."""
        llm = ChatGroq(
            temperature=0.3,
            model_name="deepseek-r1-distill-llama-70b",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Updated memory configuration with output_key
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            output_key='answer',  # Specify which key to store in memory
            return_messages=True
        )

        # System prompt
        system_prompt = """You are a helpful PDF document assistant. When answering questions:
        1. Always cite the source document and page number
        2. Keep responses clear and concise
        3. If information is not found in the documents, say so
        4. Suggest 2-3 relevant follow-up questions
        5. Format responses for readability using markdown
        """

        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(
                search_kwargs={"k": 3}
            ),
            memory=memory,
            verbose=True,
            return_source_documents=True
        )

    def process_documents(self, pdf_docs):
        """Process documents with progress tracking."""
        with st.spinner("Processing Documents..."):
            try:
                raw_text, docs_info = self.extract_pdf_text(pdf_docs)
                st.session_state.processed_docs = docs_info
                
                text_chunks = self.create_text_chunks(raw_text)
                vector_store = self.create_vector_store(text_chunks)
                st.session_state.conversation = self.initialize_chat_model(vector_store)
                
                doc_stats = [f"{name} ({pages} pages)" for name, pages in docs_info.items()]
                st.success("📚 Processed documents:\n" + "\n".join(doc_stats))
                
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")

    def format_response(self, response):
        """Format the chatbot response with source citations and suggestions."""
        answer = response['answer']
        formatted_response = f"{answer}\n\n"
        
        return formatted_response

    def render_ui(self):
        """Render the improved user interface."""
        st.set_page_config(layout="wide", page_title="📚 Smart PDF Chat")
        
        st.title("📚 Smart PDF Chat")
        st.write("Upload PDFs and ask questions about their content!")

        with st.sidebar:
            st.header("📂 Document Upload")
            pdf_docs = st.file_uploader(
                "Upload your PDF files",
                type=['pdf'],
                accept_multiple_files=True,
                help="Select one or more PDF files to analyze"
            )
            
            if st.button("Process Documents", use_container_width=True):
                if pdf_docs:
                    self.process_documents(pdf_docs)
                else:
                    st.warning("⚠️ Please upload at least one PDF file.")
            
            if st.session_state.processed_docs:
                st.subheader("📑 Processed Documents")
                for doc_name, pages in st.session_state.processed_docs.items():
                    st.text(f"📄 {doc_name} ({pages} pages)")

        if 'conversation' in st.session_state and st.session_state.conversation:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            user_query = st.chat_input("Ask a question about your documents...")
            if user_query:
                st.chat_message("user").markdown(user_query)
                
                with st.spinner("Thinking..."):
                    response = st.session_state.conversation({'question': user_query})
                    formatted_response = self.format_response(response)
                
                st.chat_message("assistant").markdown(formatted_response)
                
                st.session_state.chat_history.extend([
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": formatted_response}
                ])
        else:
            st.info("👆 Please upload and process your PDF documents to start chatting!")

def main():
    chatbot = PDFChatbot()
    chatbot.render_ui()

if __name__ == "__main__":
    main()