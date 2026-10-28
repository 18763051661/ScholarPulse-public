"""LangChain LCEL Chain: extract_chain"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..lc_model import RouterChatModel

extract_chain_chain = (
    ChatPromptTemplate.from_messages([("system","extract_chain"),("human","{input}")])
    | RouterChatModel()
    | StrOutputParser()
)
