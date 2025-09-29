from PyPDF2 import PdfReader
from langchain_core.documents import Document
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentLoader:
    def __init__(self, uploaded_file: UploadedFile):
        self.pdf_reader = PdfReader(uploaded_file)
        self.docs = self._extract_documents()
        self.doc_chunks = self._split_documents()

    def _extract_documents(self):
        docs = []
        for i, page in enumerate(self.pdf_reader.pages):
            text = page.extract_text()
            if text:
                docs.append(Document(
                    page_content=text,
                    metadata={"page": i+1}
                ))
        return docs
    
    def _split_documents(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return splitter.split_documents(self.docs)
    

