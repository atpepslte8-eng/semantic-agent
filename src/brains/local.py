"""
Local Brain Providers - Run models on your own hardware.

Supported:
- Ollama (easiest)
- llama.cpp server
- LM Studio
- vLLM
- text-generation-webui
- LocalAI
"""

import os
import json
import requests
from typing import List, Iterator, Optional
from .base import Brain, BrainConfig, BrainResponse, BrainCapability


class OllamaBrain(Brain):
    """
    Ollama - The easiest way to run local models.
    
    Setup:
        brew install ollama
        ollama pull llama3.2
    
    Models: llama3.2, mistral, codellama, phi3, gemma2, deepseek-coder-v2, etc.
    """
    
    def __init__(self, config: BrainConfig):
        super().__init__(config)
        
        self.base_url = config.base_url or "http://localhost:11434"
        
        # Ollama supports streaming and embeddings
        self._capabilities = [
            BrainCapability.TEXT,
            BrainCapability.STREAMING,
            BrainCapability.EMBEDDING
        ]
        
        # Some models support vision
        vision_models = ["llava", "bakllava", "moondream"]
        if any(vm in config.model.lower() for vm in vision_models):
            self._capabilities.append(BrainCapability.VISION)
    
    def think(self, prompt: str, system: str = None) -> str:
        response = self.think_full(prompt, system)
        return response.content
    
    def think_full(self, prompt: str, system: str = None) -> BrainResponse:
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
                "top_p": self.config.top_p,
            }
        }
        
        if system:
            payload["system"] = system
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        
        data = response.json()
        
        return BrainResponse(
            content=data["response"],
            model=data.get("model", self.config.model),
            tokens_in=data.get("prompt_eval_count", 0),
            tokens_out=data.get("eval_count", 0),
            raw=data
        )
    
    def stream(self, prompt: str, system: str = None) -> Iterator[str]:
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            }
        }
        
        if system:
            payload["system"] = system
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            stream=True,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "response" in data:
                    yield data["response"]
    
    def embed(self, text: str) -> List[float]:
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={
                "model": self.config.model,
                "prompt": text
            },
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()["embedding"]
    
    def see(self, image_path: str, prompt: str = "Describe this image") -> str:
        import base64
        
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.config.model,
                "prompt": prompt,
                "images": [image_data],
                "stream": False
            },
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()["response"]
    
    def list_models(self) -> List[str]:
        """List available Ollama models."""
        response = requests.get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]


class LlamaCppBrain(Brain):
    """
    llama.cpp server - Direct GGUF model inference.
    
    Setup:
        # Build llama.cpp
        git clone https://github.com/ggerganov/llama.cpp
        cd llama.cpp && make
        
        # Run server
        ./server -m model.gguf -c 4096
    """
    
    def __init__(self, config: BrainConfig):
        super().__init__(config)
        
        self.base_url = config.base_url or "http://localhost:8080"
        
        self._capabilities = [
            BrainCapability.TEXT,
            BrainCapability.STREAMING,
            BrainCapability.EMBEDDING
        ]
    
    def think(self, prompt: str, system: str = None) -> str:
        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"
        
        response = requests.post(
            f"{self.base_url}/completion",
            json={
                "prompt": full_prompt,
                "n_predict": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "stream": False
            },
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()["content"]
    
    def think_full(self, prompt: str, system: str = None) -> BrainResponse:
        return BrainResponse(content=self.think(prompt, system))
    
    def stream(self, prompt: str, system: str = None) -> Iterator[str]:
        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"
        
        response = requests.post(
            f"{self.base_url}/completion",
            json={
                "prompt": full_prompt,
                "n_predict": self.config.max_tokens,
                "stream": True
            },
            stream=True,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode()
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if "content" in data:
                        yield data["content"]
    
    def embed(self, text: str) -> List[float]:
        response = requests.post(
            f"{self.base_url}/embedding",
            json={"content": text},
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()["embedding"]


class LMStudioBrain(Brain):
    """
    LM Studio - GUI for local models, exposes OpenAI-compatible API.
    
    Setup:
        1. Download from lmstudio.ai
        2. Load a model
        3. Start local server (default: localhost:1234)
    """
    
    def __init__(self, config: BrainConfig):
        super().__init__(config)
        
        self.base_url = config.base_url or "http://localhost:1234/v1"
        
        self._capabilities = [
            BrainCapability.TEXT,
            BrainCapability.STREAMING
        ]
    
    def think(self, prompt: str, system: str = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "stream": False
            },
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def think_full(self, prompt: str, system: str = None) -> BrainResponse:
        return BrainResponse(content=self.think(prompt, system))


class VLLMBrain(Brain):
    """
    vLLM - High-throughput LLM serving.
    
    Setup:
        pip install vllm
        python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3-8B
    """
    
    def __init__(self, config: BrainConfig):
        super().__init__(config)
        
        self.base_url = config.base_url or "http://localhost:8000/v1"
        
        self._capabilities = [
            BrainCapability.TEXT,
            BrainCapability.STREAMING
        ]
    
    def think(self, prompt: str, system: str = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
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


class LocalAIBrain(Brain):
    """
    LocalAI - Drop-in OpenAI replacement, runs on CPU/GPU.
    
    Setup:
        docker run -p 8080:8080 localai/localai
    """
    
    def __init__(self, config: BrainConfig):
        super().__init__(config)
        
        self.base_url = config.base_url or "http://localhost:8080/v1"
        
        self._capabilities = [
            BrainCapability.TEXT,
            BrainCapability.EMBEDDING
        ]
    
    def think(self, prompt: str, system: str = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
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
    
    def embed(self, text: str) -> List[float]:
        response = requests.post(
            f"{self.base_url}/embeddings",
            json={
                "model": self.config.model,
                "input": text
            },
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
