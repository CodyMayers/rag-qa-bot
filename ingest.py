from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma

# Load document
loader = PyPDFLoader("data/2025 Benefits Guide - Consultants.pdf")
docs = loader.load()
print('✅ Loaded document')

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"✅ Split document into {len(chunks)} chunks")

# Convert to embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Save into Vector Database (Chroma)
db = Chroma(persist_directory="vector_db", embedding_function=embeddings)
db.add_documents(chunks)

print(f"✅ Ingested {len(chunks)} chunks into vector DB")