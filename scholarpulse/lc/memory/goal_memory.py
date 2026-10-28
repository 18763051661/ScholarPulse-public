"""LangChain Memory: goal_memory"""
from langchain_core.memory import BaseMemory

class GoalMemory(BaseMemory):
    def load_memory_variables(self, inputs):
        return {"goal_memory": []}
    def save_context(self, inputs, outputs):
        pass
    def clear(self):
        pass
