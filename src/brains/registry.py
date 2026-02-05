"""
Brain Registry - Get any brain by name.

Usage:
    brain = get_brain("ollama/llama3.2")
    brain = get_brain("openai/gpt-4")
    brain = get_brain("anthropic/claude-3-opus")
    brain = get_brain("http://localhost:8080/v1", model="custom")
"""

from typing import Dict, Type, Optional, List
from .base import Brain, BrainConfig, CompositeBrain

# Import all brain implementations
from .local import (
    OllamaBrain,
    LlamaCppBrain,
    LMStudioBrain,
    VLLMBrain,
    LocalAIBrain
)
from .api import (
    OpenAIBrain,
    AnthropicBrain,
    GoogleBrain,
    GroqBrain,
    DeepSeekBrain,
    OpenRouterBrain,
    OpenAICompatibleBrain
)


# Provider -> Brain class mapping
BRAIN_REGISTRY: Dict[str, Type[Brain]] = {
    # Local
    "ollama": OllamaBrain,
    "llama.cpp": LlamaCppBrain,
    "llamacpp": LlamaCppBrain,
    "lmstudio": LMStudioBrain,
    "lm-studio": LMStudioBrain,
    "vllm": VLLMBrain,
    "localai": LocalAIBrain,
    "local-ai": LocalAIBrain,
    
    # Cloud APIs
    "openai": OpenAIBrain,
    "anthropic": AnthropicBrain,
    "claude": AnthropicBrain,
    "google": GoogleBrain,
    "gemini": GoogleBrain,
    "groq": GroqBrain,
    "deepseek": DeepSeekBrain,
    "openrouter": OpenRouterBrain,
    "or": OpenRouterBrain,
}

# Default models for each provider
DEFAULT_MODELS: Dict[str, str] = {
    "ollama": "llama3.2",
    "llama.cpp": "default",
    "lmstudio": "local-model",
    "vllm": "default",
    "localai": "gpt-3.5-turbo",
    "openai": "gpt-4o",
    "anthropic": "claude-3-5-sonnet-20241022",
    "claude": "claude-3-5-sonnet-20241022",
    "google": "gemini-1.5-pro",
    "gemini": "gemini-1.5-pro",
    "groq": "llama-3.2-90b-text-preview",
    "deepseek": "deepseek-chat",
    "openrouter": "anthropic/claude-3.5-sonnet",
}


def get_brain(
    spec: str,
    model: str = None,
    api_key: str = None,
    **kwargs
) -> Brain:
    """
    Get a brain by spec string.
    
    Formats:
        "provider/model"     - e.g., "ollama/llama3.2"
        "provider"           - uses default model
        "http://..."         - OpenAI-compatible endpoint
    
    Args:
        spec: Brain specification string
        model: Override model (optional)
        api_key: Override API key (optional)
        **kwargs: Additional config options
    
    Returns:
        Configured Brain instance
    
    Examples:
        get_brain("ollama/llama3.2")
        get_brain("openai/gpt-4")
        get_brain("anthropic")  # uses default claude model
        get_brain("http://localhost:8080/v1", model="my-model")
    """
    
    # Check if it's a URL (OpenAI-compatible endpoint)
    if spec.startswith("http://") or spec.startswith("https://"):
        config = BrainConfig(
            provider="openai-compatible",
            model=model or "default",
            base_url=spec,
            api_key=api_key,
            **kwargs
        )
        return OpenAICompatibleBrain(config)
    
    # Parse provider/model
    if "/" in spec:
        provider, model_name = spec.split("/", 1)
        if model:
            model_name = model  # Override if provided
    else:
        provider = spec
        model_name = model or DEFAULT_MODELS.get(provider.lower(), "default")
    
    provider = provider.lower()
    
    # Look up brain class
    if provider not in BRAIN_REGISTRY:
        # Try as OpenAI-compatible with base_url
        if kwargs.get("base_url"):
            config = BrainConfig(
                provider=provider,
                model=model_name,
                api_key=api_key,
                **kwargs
            )
            return OpenAICompatibleBrain(config)
        
        raise ValueError(
            f"Unknown provider: {provider}\n"
            f"Available: {list(BRAIN_REGISTRY.keys())}"
        )
    
    brain_class = BRAIN_REGISTRY[provider]
    
    config = BrainConfig(
        provider=provider,
        model=model_name,
        api_key=api_key,
        **kwargs
    )
    
    return brain_class(config)


def list_brains() -> Dict[str, List[str]]:
    """
    List available brain providers and their default models.
    
    Returns:
        Dict of provider -> default model
    """
    return {
        "local": {
            "ollama": "llama3.2, mistral, codellama, etc.",
            "llama.cpp": "Any GGUF model",
            "lmstudio": "Any loaded model",
            "vllm": "Any HuggingFace model",
            "localai": "Any supported model",
        },
        "cloud": {
            "openai": "gpt-4o, gpt-4-turbo, o1, etc.",
            "anthropic": "claude-3-opus, claude-3-sonnet, etc.",
            "google": "gemini-1.5-pro, gemini-1.5-flash",
            "groq": "llama-3.2-90b, mixtral-8x7b",
            "deepseek": "deepseek-chat, deepseek-coder",
            "openrouter": "100+ models via one API",
        }
    }


def register_brain(name: str, brain_class: Type[Brain], default_model: str = None):
    """
    Register a custom brain provider.
    
    Args:
        name: Provider name (e.g., "mybrain")
        brain_class: Brain subclass
        default_model: Default model for this provider
    """
    BRAIN_REGISTRY[name.lower()] = brain_class
    if default_model:
        DEFAULT_MODELS[name.lower()] = default_model


def get_fallback_brain(specs: List[str], **kwargs) -> CompositeBrain:
    """
    Get a composite brain that falls back through multiple providers.
    
    Args:
        specs: List of brain specs to try in order
    
    Example:
        brain = get_fallback_brain([
            "ollama/llama3.2",   # Try local first
            "groq/llama-3.2-90b",  # Fast cloud fallback
            "openai/gpt-4o"     # Premium fallback
        ])
    """
    brains = [get_brain(spec, **kwargs) for spec in specs]
    return CompositeBrain(brains, strategy="fallback")


# Convenience aliases
def ollama(model: str = "llama3.2", **kwargs) -> Brain:
    """Quick access to Ollama brain."""
    return get_brain(f"ollama/{model}", **kwargs)


def openai(model: str = "gpt-4o", **kwargs) -> Brain:
    """Quick access to OpenAI brain."""
    return get_brain(f"openai/{model}", **kwargs)


def anthropic(model: str = "claude-3-5-sonnet-20241022", **kwargs) -> Brain:
    """Quick access to Anthropic brain."""
    return get_brain(f"anthropic/{model}", **kwargs)


def groq(model: str = "llama-3.2-90b-text-preview", **kwargs) -> Brain:
    """Quick access to Groq brain."""
    return get_brain(f"groq/{model}", **kwargs)


def local(base_url: str, model: str = "default", **kwargs) -> Brain:
    """Quick access to any local OpenAI-compatible server."""
    return get_brain(base_url, model=model, **kwargs)
