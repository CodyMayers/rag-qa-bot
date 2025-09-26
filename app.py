import streamlit as st
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain_core.documents import Document

# Load vector DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="vector_db", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

# Connect to Ollama LLM
llm = ChatOllama(model="gpt-oss:20b") # to use any other LLM available on Ollama, just change the model name here

qa = RetrievalQA.from_llm(llm=llm, retriever=retriever)

# Streamlit UI
st.title("📚 Document Q&A Bot")
uploaded_file = st.file_uploader("Upload Document", type=["pdf"])
if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    # Extract all text per page and wrap as Documents
    docs = []
    for i, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if text:
            docs.append(Document(
                page_content=text, 
                metadata={"page": i+1}
            ))

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Convert to embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Save into Vector Database (Chroma)
    db = Chroma(persist_directory="vector_db", embedding_function=embeddings)
    db.add_documents(chunks)

query = st.text_input("Ask a question about your document:")
if query:
    answer = qa.invoke(query)
    st.write(answer['result'])