"""LangChain Prompt: concept_extract"""
from langchain_core.prompts import ChatPromptTemplate

CONCEPT_EXTRACT = ChatPromptTemplate.from_messages([
    ("system", "你是学情分析专家(concept_extract)。"),
    ("human", "{input}"),
])
