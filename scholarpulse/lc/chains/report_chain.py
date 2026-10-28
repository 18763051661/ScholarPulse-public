"""LangChain LCEL Chain: report_chain"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..lc_model import RouterChatModel

report_chain_chain = (
    ChatPromptTemplate.from_messages([("system","report_chain"),("human","{input}")])
    | RouterChatModel()
    | StrOutputParser()
)
