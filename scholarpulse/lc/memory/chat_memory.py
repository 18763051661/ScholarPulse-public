"""LangChain Memory: chat_memory"""
from langchain_core.memory import BaseMemory

class ChatMemory(BaseMemory):
    def load_memory_variables(self, inputs):
        return {"chat_memory": []}
    def save_context(self, inputs, outputs):
        pass
    def clear(self):
        pass
