"""LangChain Memory: feedback_memory"""
from langchain_core.memory import BaseMemory

class FeedbackMemory(BaseMemory):
    def load_memory_variables(self, inputs):
        return {"feedback_memory": []}
    def save_context(self, inputs, outputs):
        pass
    def clear(self):
        pass
