"""LangChain LCEL Chain: error_chain"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..lc_model import RouterChatModel

error_chain_chain = (
    ChatPromptTemplate.from_messages([("system","error_chain"),("human","{input}")])
    | RouterChatModel()
    | StrOutputParser()
)
