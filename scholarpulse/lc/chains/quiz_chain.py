"""LangChain LCEL Chain: quiz_chain"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..lc_model import RouterChatModel

quiz_chain_chain = (
    ChatPromptTemplate.from_messages([("system","quiz_chain"),("human","{input}")])
    | RouterChatModel()
    | StrOutputParser()
)
