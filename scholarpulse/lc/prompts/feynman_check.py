"""LangChain Prompt: feynman_check"""
from langchain_core.prompts import ChatPromptTemplate

FEYNMAN_CHECK = ChatPromptTemplate.from_messages([
    ("system", "你是学情分析专家(feynman_check)。"),
    ("human", "{input}"),
])
