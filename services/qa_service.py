from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA


class QAService:
    def __init__(self, retriever):
        self.llm = ChatOllama(model="gpt-oss:20b")
        self.qa = RetrievalQA.from_llm(llm=self.llm, retriever=retriever)

    def get_answer(self, query):
        return self.qa.invoke(query)['result']