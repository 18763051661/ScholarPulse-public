"""LangChain Prompt: report_writer"""
from langchain_core.prompts import ChatPromptTemplate

REPORT_WRITER = ChatPromptTemplate.from_messages([
    ("system", "你是学情分析专家(report_writer)。"),
    ("human", "{input}"),
])
