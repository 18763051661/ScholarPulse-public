"""LangChain Prompt: noise_filter"""
from langchain_core.prompts import ChatPromptTemplate

NOISE_FILTER = ChatPromptTemplate.from_messages([
    ("system", "你是学情分析专家(noise_filter)。"),
    ("human", "{input}"),
])
