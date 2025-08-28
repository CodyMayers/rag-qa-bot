import streamlit as st
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma

# Load vector DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="vector_db", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

# Connect to Ollama LLM
llm = ChatOllama(model="mistral") # to use any other LLM available on Ollama, just change the model name here

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Streamlit UI
st.title("📚 Document Q&A Bot")
query = st.text_input("Ask a question about your document:")
if query:
    answer = qa.invoke(query)
    st.write(answer['result'])