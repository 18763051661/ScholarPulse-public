"""LangChain Prompt: quiz_gen"""
from langchain_core.prompts import ChatPromptTemplate

QUIZ_GEN = ChatPromptTemplate.from_messages([
    ("system", "你是学情分析专家(quiz_gen)。"),
    ("human", "{input}"),
])
