"""LangChain Retriever: example_retriever"""
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

class ExampleRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager=None):
        return [Document(page_content=f"example_retriever: " + query, metadata={"src": "example_retriever"})]
