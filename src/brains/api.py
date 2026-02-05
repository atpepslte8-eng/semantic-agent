"""
API Brain Providers - Cloud-hosted models.

Supported:
- OpenAI (GPT-4, GPT-4o, o1, etc.)
- Anthropic (Claude 3, Claude 3.5)
- Google (Gemini)
- Groq (fast inference)
- Together AI
- Mistral
- DeepSeek
- OpenRouter (access many via one API)
- Any OpenAI-compatible endpoint
"""

import os
import json
import requests
from typing import List, Iterator, Optional
from .base import Brain, BrainConfig, BrainResponse, BrainCapability


class OpenAIBrain(Brain):
    """
    OpenAI - GPT-4, GPT-4o, o1, etc.
    
    Setup:
        export OPENAI_API_KEY=sk-...
    """
    
    def __init__(self, config: BrainConfig):
        super().__init__(config)
        
        self.base_url = config.base_url or "https://api.openai.com/v1"
        self.api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key required (set OPENAI_API_KEY)")
        
        self._capabilities = [
            BrainCapability.TEXT,
            BrainCapability.STREAMING,
            BrainCapability.EMBEDDING,
            BrainCapability.FUNCTION
        ]
        
        # Vision models
        if any(v in config.model for v in ["gpt-4o", "gpt-4-vision", "gpt-4-turbo"]):
            self._capabilities.append(BrainCapability.VISION)
    
    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def think(self, prompt: str, system: str = None) -> str:
        response = self.think_full(prompt, system)
        return response.content
    
    def think_full(self, prompt: str, system: str = None) -> BrainResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json={
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            },
            timeout=self.config.timeout
        )
        response.raise_for_status()
        
        data = response.json()
        usage = data.get("usage", {})
        
        return BrainResponse(
            content=data["choices"][0]["message"]["content"],
            model=data.get("model", self.config.model),
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
            raw=data
        )
    
    def stream(self, prompt: str, system: str = None) -> Iterator[str]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json={
                "model": self.config.model,
                "messages": messages,
                "stream": True
            },
            stream=True,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode()
                if line.startswith("data: ") and line != "data: [DONE]":
                    data = json.loads(line[6:])
                    delta = data["choices"][0].get("delta", {})
                    if "content" in delta:
                        yield delta["content"]
    
    def embed(self, text: str) -> List[float]:
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=self._headers(),
            json={
                "model": "text-embedding-3-small",
                "input": text
            },
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=self._headers(),
            json={
                "model": "text-embedding-3-small",
                "input": texts
            },
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return [d["embedding"] for d in response.json()["data"]]


class AnthropicBrain(Brain):
    """
    Anthropic - Claude 3, Claude 3.5, Claude 4.
    
    Setup:
        export ANTHROPIC_API_KEY=sk-ant-...
    """
    
    def __init__(self, config: BrainConfig):
        super().__init__(config)
        
        self.base_url = config.base_url or "https://api.anthropic.com"
        self.api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("Anthropic API key required (set ANTHROPIC_API_KEY)")
        
        self._capabilities = [
            BrainCapability.TEXT,
            BrainCapability.STREAMING,
            BrainCapability.VISION
        ]
    
    def _headers(self):
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
    
    def think(self, prompt: str, system: str = None) -> str:
        response = self.think_full(prompt, system)
        return response.content
    
    def think_full(self, prompt: str, system: str = None) -> BrainResponse:
        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if system:
            payload["system"] = system
        
        response = requests.post(
            f"{self.base_url}/v1/messages",
            headers=self._headers(),
            json=payload,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        
        data = response.json()
        usage = data.get("usage", {})
        
        return BrainResponse(
            content=data["content"][0]["text"],
            model=data.get("model", self.config.model),
            tokens_in=usage.get("input_tokens", 0),
            tokens_out=usage.get("output_tokens", 0),
            raw=data
        )
    
    def stream(self, prompt: str, system: str = None) -> Iterator[str]:
        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }
        
        if system:
            payload["system"] = system
        
        response = requests.post(
            f"{self.base_url}/v1/messages",
            headers=self._headers(),
            json=payload,
            stream=True,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode()
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if data["type"] == "content_block_delta":
                        yield data["delta"].get("text", "")


class GoogleBrain(Brain):
    """
    Google Gemini - Gemini Pro, Gemini Ultra.
    
    Setup:
        export GOOGLE_API_KEY=...
    """
    
    def __init__(self, config: BrainConfig):
        super().__init__(config)
        
        self.api_key = config.api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Google API key required (set GOOGLE_API_KEY)")
        
        self._capabilities = [
            BrainCapability.TEXT,
            BrainCapability.STREAMING,
            BrainCapability.VISION,
            BrainCapability.EMBEDDING
        ]
    
    def think(self, prompt: str, system: str = None) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.config.model}:generateContent"
        
        contents = [{"parts": [{"text": prompt}]}]
        
        payload = {"contents": contents}
        
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        
        response = requests.post(
            url,
            params={"key": self.api_key},
            json=payload,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    
    def think_full(self, prompt: str, system: str = None) -> BrainResponse:
        return BrainResponse(content=self.think(prompt, system))
    
    def embed(self, text: str) -> List[float]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"
        
        response = requests.post(
            url,
            params={"key": self.api_key},
            json={"content": {"parts": [{"text": text}]}},
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()["embedding"]["values"]


class GroqBrain(Brain):
    """
    Groq - Ultra-fast inference for open models.
    
    Setup:
        export GROQ_API_KEY=gsk_...
    
    Models: llama-3.2-90b, mixtral-8x7b, etc.
    """
    
    def __init__(self, config: BrainConfig):
        super().__init__(config)
        
        self.base_url = "https://api.groq.com/openai/v1"
        self.api_key = config.api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("Groq API key required (set GROQ_API_KEY)")
        
        self._capabilities = [
            BrainCapability.TEXT,
            BrainCapability.STREAMING
        ]
    
    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def think(self, prompt: str, system: str = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json={
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            },
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def think_full(self, prompt: str, system: str = None) -> BrainResponse:
        return BrainResponse(content=self.think(prompt, system))


class DeepSeekBrain(Brain):
    """
    DeepSeek - Strong coding and reasoning models.
    
    Setup:
        export DEEPSEEK_API_KEY=...
    """
    
    def __init__(self, config: BrainConfig):
        super().__init__(config)
        
        self.base_url = "https://api.deepseek.com/v1"
        self.api_key = config.api_key or os.getenv("DEEPSEEK_API_KEY")
        
        if not self.api_key:
            raise ValueError("DeepSeek API key required (set DEEPSEEK_API_KEY)")
        
        self._capabilities = [
            BrainCapability.TEXT,
            BrainCapability.STREAMING
        ]
    
    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def think(self, prompt: str, system: str = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json={
                "model": self.config.model,
                "messages": messages,
            },
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def think_full(self, prompt: str, system: str = None) -> BrainResponse:
        return BrainResponse(content=self.think(prompt, system))


class OpenRouterBrain(Brain):
    """
    OpenRouter - Access 100+ models through one API.
    
    Setup:
        export OPENROUTER_API_KEY=sk-or-...
    
    Models: openai/gpt-4, anthropic/claude-3, meta-llama/llama-3, etc.
    """
    
    def __init__(self, config: BrainConfig):
        super().__init__(config)
        
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_key = config.api_key or os.getenv("OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenRouter API key required (set OPENROUTER_API_KEY)")
        
        self._capabilities = [
            BrainCapability.TEXT,
            BrainCapability.STREAMING
        ]
    
    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/semantic-agent",
            "X-Title": "Semantic Agent"
        }
    
    def think(self, prompt: str, system: str = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json={
                "model": self.config.model,
                "messages": messages,
            },
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def think_full(self, prompt: str, system: str = None) -> BrainResponse:
        return BrainResponse(content=self.think(prompt, system))


class OpenAICompatibleBrain(Brain):
    """
    Generic OpenAI-compatible endpoint.
    
    Works with: vLLM, LocalAI, LiteLLM, FastChat, etc.
    """
    
    def __init__(self, config: BrainConfig):
        super().__init__(config)
        
        if not config.base_url:
            raise ValueError("base_url required for OpenAI-compatible brain")
        
        self.base_url = config.base_url
        self.api_key = config.api_key or "not-needed"
        
        self._capabilities = [
            BrainCapability.TEXT,
            BrainCapability.STREAMING
        ]
    
    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def think(self, prompt: str, system: str = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json={
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            },
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def think_full(self, prompt: str, system: str = None) -> BrainResponse:
        return BrainResponse(content=self.think(prompt, system))
