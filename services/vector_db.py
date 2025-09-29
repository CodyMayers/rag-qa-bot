from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class VectorDB:
    def __init__(self, persist_dir="vector_db", embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.db = Chroma(persist_directory=persist_dir, embedding_function=self.embeddings)

    def add_docs(self, docs):
        self.db.add_documents(docs)

    def get_retriever(self, k=3):
        return self.db.as_retriever(search_kwargs={"k": k})
