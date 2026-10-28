"""LangChain Memory: session_memory"""
from langchain_core.memory import BaseMemory

class SessionMemory(BaseMemory):
    def load_memory_variables(self, inputs):
        return {"session_memory": []}
    def save_context(self, inputs, outputs):
        pass
    def clear(self):
        pass
