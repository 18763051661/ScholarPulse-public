"""LangChain Prompt: root_cause"""
from langchain_core.prompts import ChatPromptTemplate

ROOT_CAUSE = ChatPromptTemplate.from_messages([
    ("system", "你是学情分析专家(root_cause)。"),
    ("human", "{input}"),
])
