"""LangChain Prompt: mastery_advice"""
from langchain_core.prompts import ChatPromptTemplate

MASTERY_ADVICE = ChatPromptTemplate.from_messages([
    ("system", "你是学情分析专家(mastery_advice)。"),
    ("human", "{input}"),
])
