"""LangChain Prompt: error_diagnose"""
from langchain_core.prompts import ChatPromptTemplate

ERROR_DIAGNOSE = ChatPromptTemplate.from_messages([
    ("system", "你是学情分析专家(error_diagnose)。"),
    ("human", "{input}"),
])
