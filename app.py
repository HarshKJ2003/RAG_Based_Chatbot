import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

def process_documents(uploaded_files):
    """Process uploaded PDF/TXT files into text chunks."""
    texts = []
    for file in uploaded_files:
        if file.type == "application/pdf":
            pdf_reader = PdfReader(file)
            text = "\n".join([page.extract_text() for page in pdf_reader.pages])
            texts.append(text)
        elif file.type == "text/plain":
            text = file.read().decode("utf-8")
            texts.append(text)
    return texts

def main():
    st.title("📄 RAG Chatbot (Free LLM)")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload documents", 
        type=["pdf", "txt"], 
        accept_multiple_files=True,
        help="Upload PDF/TXT files to create your knowledge base"
    )
    
    # Document processing
    if uploaded_files and not st.session_state.get("vector_store"):
        with st.spinner("Processing documents..."):
            texts = process_documents(uploaded_files)
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_text("\n\n".join(texts))
            
            # Create embeddings and vector store
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vector_store = FAISS.from_texts(splits, embeddings)
            st.session_state.vector_store = vector_store
            st.success("Documents processed! You can now ask questions.")

    # Question answering
    if "vector_store" in st.session_state:
        # Initialize free LLM
        with st.spinner("Loading AI model..."):
            hf_pipe = pipeline(
                "text2text-generation",
                model="google/flan-t5-small",
                max_length=512
            )
            llm = HuggingFacePipeline(pipeline=hf_pipe)
            
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=st.session_state.vector_store.as_retriever()
            )

        # Chat interface
        user_question = st.chat_input("Ask a question about your documents:")
        if user_question:
            with st.spinner("Thinking..."):
                response = qa_chain.run(user_question)
                st.chat_message("user").write(user_question)
                st.chat_message("assistant").write(response)

if __name__ == "__main__":
    main()