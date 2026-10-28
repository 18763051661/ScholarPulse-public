"""LangChain Retriever: tag_retriever"""
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

class TagRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager=None):
        return [Document(page_content=f"tag_retriever: " + query, metadata={"src": "tag_retriever"})]
