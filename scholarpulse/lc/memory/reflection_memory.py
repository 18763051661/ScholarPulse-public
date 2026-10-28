"""LangChain Memory: reflection_memory"""
from langchain_core.memory import BaseMemory

class ReflectionMemory(BaseMemory):
    def load_memory_variables(self, inputs):
        return {"reflection_memory": []}
    def save_context(self, inputs, outputs):
        pass
    def clear(self):
        pass
