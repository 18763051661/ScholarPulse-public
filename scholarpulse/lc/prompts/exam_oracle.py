"""LangChain Prompt: exam_oracle"""
from langchain_core.prompts import ChatPromptTemplate

EXAM_ORACLE = ChatPromptTemplate.from_messages([
    ("system", "你是学情分析专家(exam_oracle)。"),
    ("human", "{input}"),
])
