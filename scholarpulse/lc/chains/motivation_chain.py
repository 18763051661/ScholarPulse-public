"""LangChain LCEL Chain: motivation_chain"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..lc_model import RouterChatModel

motivation_chain_chain = (
    ChatPromptTemplate.from_messages([("system","motivation_chain"),("human","{input}")])
    | RouterChatModel()
    | StrOutputParser()
)
