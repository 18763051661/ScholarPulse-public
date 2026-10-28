"""LangChain Retriever: weakness_retriever"""
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

class WeaknessRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager=None):
        return [Document(page_content=f"weakness_retriever: " + query, metadata={"src": "weakness_retriever"})]
