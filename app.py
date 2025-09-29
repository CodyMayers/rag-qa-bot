import streamlit as st
from services.vector_db import VectorDB
from services.document_loader import DocumentLoader
from services.qa_service import QAService


# Streamlit UI
st.title("📚 Document Q&A Bot")

# Initialize vector DB
db = VectorDB()

# Upload Documents
uploaded_file = st.file_uploader("Upload Document", type=["pdf"])
if uploaded_file:
    # Read file, extract text, and split into chunks
    doc_loader = DocumentLoader(uploaded_file)

    # Save into Vector Database
    db.add_docs(doc_loader.doc_chunks)

query = st.text_input("Ask a question about your document:")
if query:
    retriever = db.get_retriever(k=3)
    qa = QAService(retriever=retriever)
    answer = qa.get_answer(query)
    st.write(answer)