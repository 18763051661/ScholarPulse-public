"""LangChain Memory: mastery_memory"""
from langchain_core.memory import BaseMemory

class MasteryMemory(BaseMemory):
    def load_memory_variables(self, inputs):
        return {"mastery_memory": []}
    def save_context(self, inputs, outputs):
        pass
    def clear(self):
        pass
