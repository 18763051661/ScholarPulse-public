"""LangChain Prompt: grounding_check"""
from langchain_core.prompts import ChatPromptTemplate

GROUNDING_CHECK = ChatPromptTemplate.from_messages([
    ("system", "你是学情分析专家(grounding_check)。"),
    ("human", "{input}"),
])
