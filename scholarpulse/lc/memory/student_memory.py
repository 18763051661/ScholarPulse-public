"""LangChain Memory: student_memory"""
from langchain_core.memory import BaseMemory

class StudentMemory(BaseMemory):
    def load_memory_variables(self, inputs):
        return {"student_memory": []}
    def save_context(self, inputs, outputs):
        pass
    def clear(self):
        pass
