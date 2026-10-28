"""LangChain Prompt: confusion_pair"""
from langchain_core.prompts import ChatPromptTemplate

CONFUSION_PAIR = ChatPromptTemplate.from_messages([
    ("system", "你是学情分析专家(confusion_pair)。"),
    ("human", "{input}"),
])
