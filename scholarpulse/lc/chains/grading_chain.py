"""LangChain LCEL Chain: grading_chain"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..lc_model import RouterChatModel

grading_chain_chain = (
    ChatPromptTemplate.from_messages([("system","grading_chain"),("human","{input}")])
    | RouterChatModel()
    | StrOutputParser()
)
