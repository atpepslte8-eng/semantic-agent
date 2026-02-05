"""
Brain Providers - Plug any LLM into the Semantic Agent.

Supported:
- Local: Ollama, llama.cpp, LM Studio, vLLM, text-generation-webui
- API: OpenAI, Anthropic, Google, Groq, Together, Mistral, DeepSeek
- Custom: Any OpenAI-compatible endpoint

Usage:
    from brains import get_brain, list_brains
    
    brain = get_brain("ollama/llama3.2")
    response = brain.think("Hello")
"""

from .base import Brain, BrainConfig
from .registry import get_brain, list_brains, register_brain

__all__ = [
    "Brain",
    "BrainConfig", 
    "get_brain",
    "list_brains",
    "register_brain"
]
