"""LangChain LCEL Chain: verify_chain"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..lc_model import RouterChatModel

verify_chain_chain = (
    ChatPromptTemplate.from_messages([("system","verify_chain"),("human","{input}")])
    | RouterChatModel()
    | StrOutputParser()
)
