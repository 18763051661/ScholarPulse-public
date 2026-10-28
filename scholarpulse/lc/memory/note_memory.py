"""LangChain Memory: note_memory"""
from langchain_core.memory import BaseMemory

class NoteMemory(BaseMemory):
    def load_memory_variables(self, inputs):
        return {"note_memory": []}
    def save_context(self, inputs, outputs):
        pass
    def clear(self):
        pass
