"""LangChain LCEL Chain: rootcause_chain"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..lc_model import RouterChatModel

rootcause_chain_chain = (
    ChatPromptTemplate.from_messages([("system","rootcause_chain"),("human","{input}")])
    | RouterChatModel()
    | StrOutputParser()
)
