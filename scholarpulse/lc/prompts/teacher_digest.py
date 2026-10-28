"""LangChain Prompt: teacher_digest"""
from langchain_core.prompts import ChatPromptTemplate

TEACHER_DIGEST = ChatPromptTemplate.from_messages([
    ("system", "你是学情分析专家(teacher_digest)。"),
    ("human", "{input}"),
])
