"""LangChain Retriever: history_retriever"""
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

class HistoryRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager=None):
        return [Document(page_content=f"history_retriever: " + query, metadata={"src": "history_retriever"})]
