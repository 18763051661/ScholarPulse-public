"""LangChain LCEL Chain: flashcard_chain"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..lc_model import RouterChatModel

flashcard_chain_chain = (
    ChatPromptTemplate.from_messages([("system","flashcard_chain"),("human","{input}")])
    | RouterChatModel()
    | StrOutputParser()
)
