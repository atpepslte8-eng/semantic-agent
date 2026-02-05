"""
Base Brain Interface - All brains implement this.

A Brain is just: text in → text out
With optional: embeddings, streaming, vision
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Iterator, Dict, Any, Callable
from enum import Enum


class BrainCapability(Enum):
    """What can a brain do?"""
    TEXT = "text"           # Basic text generation
    EMBEDDING = "embedding" # Generate embeddings
    STREAMING = "streaming" # Stream responses
    VISION = "vision"       # Process images
    FUNCTION = "function"   # Function/tool calling


@dataclass
class BrainConfig:
    """Configuration for a brain."""
    
    # Identity
    provider: str           # e.g., "ollama", "openai", "anthropic"
    model: str              # e.g., "llama3.2", "gpt-4", "claude-3"
    
    # Connection
    base_url: Optional[str] = None    # API endpoint
    api_key: Optional[str] = None     # API key (from env if not set)
    
    # Generation settings
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    
    # Timeouts
    timeout: int = 120      # seconds
    
    # Extra provider-specific settings
    extra: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def full_name(self) -> str:
        return f"{self.provider}/{self.model}"


@dataclass
class BrainResponse:
    """Response from a brain."""
    content: str
    
    # Metadata
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    
    # For streaming
    is_complete: bool = True
    
    # Raw response for debugging
    raw: Optional[Dict[str, Any]] = None


class Brain(ABC):
    """
    Abstract base class for all brains.
    
    A brain is a text-in, text-out interface to any LLM.
    """
    
    def __init__(self, config: BrainConfig):
        self.config = config
        self._capabilities: List[BrainCapability] = [BrainCapability.TEXT]
    
    @property
    def name(self) -> str:
        return self.config.full_name
    
    @property
    def capabilities(self) -> List[BrainCapability]:
        return self._capabilities
    
    def has_capability(self, cap: BrainCapability) -> bool:
        return cap in self._capabilities
    
    @abstractmethod
    def think(self, prompt: str, system: str = None) -> str:
        """
        Core method: text in → text out.
        
        Args:
            prompt: The user prompt
            system: Optional system prompt
            
        Returns:
            The response text
        """
        pass
    
    @abstractmethod
    def think_full(self, prompt: str, system: str = None) -> BrainResponse:
        """
        Full response with metadata.
        """
        pass
    
    def stream(self, prompt: str, system: str = None) -> Iterator[str]:
        """
        Stream response tokens.
        
        Default: just yield the full response.
        Override for true streaming.
        """
        yield self.think(prompt, system)
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        Override if the brain supports embeddings.
        """
        raise NotImplementedError(f"{self.name} doesn't support embeddings")
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding - default is sequential."""
        return [self.embed(t) for t in texts]
    
    def see(self, image_path: str, prompt: str = "Describe this image") -> str:
        """
        Process an image.
        
        Override if the brain supports vision.
        """
        raise NotImplementedError(f"{self.name} doesn't support vision")
    
    def ping(self) -> bool:
        """Check if the brain is reachable."""
        try:
            response = self.think("Say 'ok' in one word.")
            return len(response) > 0
        except:
            return False
    
    def __call__(self, prompt: str) -> str:
        """Allow brain(prompt) syntax."""
        return self.think(prompt)
    
    def __repr__(self):
        return f"<Brain {self.name}>"


class CompositeBrain(Brain):
    """
    A brain that combines multiple brains.
    
    Use cases:
    - Fallback: try brain A, if fail try brain B
    - Routing: use different brains for different tasks
    - Ensemble: combine multiple responses
    """
    
    def __init__(self, brains: List[Brain], strategy: str = "fallback"):
        """
        Args:
            brains: List of brains to use
            strategy: "fallback", "round_robin", or "custom"
        """
        config = BrainConfig(
            provider="composite",
            model="+".join(b.config.model for b in brains)
        )
        super().__init__(config)
        
        self.brains = brains
        self.strategy = strategy
        self._current_index = 0
    
    def think(self, prompt: str, system: str = None) -> str:
        if self.strategy == "fallback":
            return self._fallback_think(prompt, system)
        elif self.strategy == "round_robin":
            return self._round_robin_think(prompt, system)
        else:
            return self._fallback_think(prompt, system)
    
    def think_full(self, prompt: str, system: str = None) -> BrainResponse:
        return BrainResponse(content=self.think(prompt, system))
    
    def _fallback_think(self, prompt: str, system: str = None) -> str:
        """Try each brain until one works."""
        errors = []
        for brain in self.brains:
            try:
                return brain.think(prompt, system)
            except Exception as e:
                errors.append(f"{brain.name}: {e}")
        
        raise RuntimeError(f"All brains failed: {errors}")
    
    def _round_robin_think(self, prompt: str, system: str = None) -> str:
        """Rotate through brains."""
        brain = self.brains[self._current_index]
        self._current_index = (self._current_index + 1) % len(self.brains)
        return brain.think(prompt, system)
