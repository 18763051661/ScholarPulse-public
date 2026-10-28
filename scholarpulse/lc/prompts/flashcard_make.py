"""LangChain Prompt: flashcard_make"""
from langchain_core.prompts import ChatPromptTemplate

FLASHCARD_MAKE = ChatPromptTemplate.from_messages([
    ("system", "你是学情分析专家(flashcard_make)。"),
    ("human", "{input}"),
])
