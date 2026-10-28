"""LangChain Retriever: note_retriever"""
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

class NoteRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager=None):
        return [Document(page_content=f"note_retriever: " + query, metadata={"src": "note_retriever"})]
