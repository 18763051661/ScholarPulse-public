"""LangChain Memory: exam_memory"""
from langchain_core.memory import BaseMemory

class ExamMemory(BaseMemory):
    def load_memory_variables(self, inputs):
        return {"exam_memory": []}
    def save_context(self, inputs, outputs):
        pass
    def clear(self):
        pass
