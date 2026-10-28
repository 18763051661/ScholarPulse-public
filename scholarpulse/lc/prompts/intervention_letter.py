"""LangChain Prompt: intervention_letter"""
from langchain_core.prompts import ChatPromptTemplate

INTERVENTION_LETTER = ChatPromptTemplate.from_messages([
    ("system", "你是学情分析专家(intervention_letter)。"),
    ("human", "{input}"),
])
