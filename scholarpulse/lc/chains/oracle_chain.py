"""LangChain LCEL Chain: oracle_chain"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..lc_model import RouterChatModel

oracle_chain_chain = (
    ChatPromptTemplate.from_messages([("system","oracle_chain"),("human","{input}")])
    | RouterChatModel()
    | StrOutputParser()
)
