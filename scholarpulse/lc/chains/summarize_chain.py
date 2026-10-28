"""LangChain LCEL Chain: summarize_chain"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..lc_model import RouterChatModel

summarize_chain_chain = (
    ChatPromptTemplate.from_messages([("system","summarize_chain"),("human","{input}")])
    | RouterChatModel()
    | StrOutputParser()
)
