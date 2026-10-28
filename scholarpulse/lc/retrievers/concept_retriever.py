"""LangChain Retriever: concept_retriever"""
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

class ConceptRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager=None):
        return [Document(page_content=f"concept_retriever: " + query, metadata={"src": "concept_retriever"})]
