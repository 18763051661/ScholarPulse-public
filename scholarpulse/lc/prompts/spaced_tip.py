"""LangChain Prompt: spaced_tip"""
from langchain_core.prompts import ChatPromptTemplate

SPACED_TIP = ChatPromptTemplate.from_messages([
    ("system", "你是学情分析专家(spaced_tip)。"),
    ("human", "{input}"),
])
