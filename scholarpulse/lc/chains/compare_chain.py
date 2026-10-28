"""LangChain LCEL Chain: compare_chain"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..lc_model import RouterChatModel

compare_chain_chain = (
    ChatPromptTemplate.from_messages([("system","compare_chain"),("human","{input}")])
    | RouterChatModel()
    | StrOutputParser()
)
