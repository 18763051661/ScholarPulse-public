"""LangChain LCEL Chain: feynman_chain"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..lc_model import RouterChatModel

feynman_chain_chain = (
    ChatPromptTemplate.from_messages([("system","feynman_chain"),("human","{input}")])
    | RouterChatModel()
    | StrOutputParser()
)
