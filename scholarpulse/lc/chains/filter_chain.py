"""LangChain LCEL Chain: filter_chain"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..lc_model import RouterChatModel

filter_chain_chain = (
    ChatPromptTemplate.from_messages([("system","filter_chain"),("human","{input}")])
    | RouterChatModel()
    | StrOutputParser()
)
