"""LangChain Memory: streak_memory"""
from langchain_core.memory import BaseMemory

class StreakMemory(BaseMemory):
    def load_memory_variables(self, inputs):
        return {"streak_memory": []}
    def save_context(self, inputs, outputs):
        pass
    def clear(self):
        pass
