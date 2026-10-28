"""LangChain Retriever: material_retriever"""
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

class MaterialRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager=None):
        return [Document(page_content=f"material_retriever: " + query, metadata={"src": "material_retriever"})]
