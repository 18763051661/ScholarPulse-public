"""LangChain Retriever: quiz_retriever"""
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

class QuizRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager=None):
        return [Document(page_content=f"quiz_retriever: " + query, metadata={"src": "quiz_retriever"})]
