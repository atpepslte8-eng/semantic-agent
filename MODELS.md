# Recommended Models for Forever Mode

Running forever means **zero API costs**. Choose a model that fits your hardware.

## Quick Recommendations

| RAM | GPU VRAM | Recommended Model | Quality |
|-----|----------|-------------------|---------|
| 8GB | None | `ollama/phi3:mini` | ⭐⭐ |
| 16GB | None | `ollama/llama3.2:3b` | ⭐⭐⭐ |
| 16GB | 8GB | `ollama/llama3.2:8b` | ⭐⭐⭐⭐ |
| 32GB | 12GB+ | `ollama/llama3.2:70b-q4` | ⭐⭐⭐⭐⭐ |
| 32GB | 24GB+ | `ollama/deepseek-coder-v2` | ⭐⭐⭐⭐⭐ (coding) |

## Model Tiers

### Tier 1: Lightweight (runs on anything)
*For: Raspberry Pi, old laptops, low-power servers*

```bash
ollama pull phi3:mini        # 2GB, surprisingly capable
ollama pull qwen2:0.5b       # Tiny but fast
ollama pull tinyllama        # 1.1B params
```

**Best for:** Simple tasks, quick responses, always-on with minimal power draw.

### Tier 2: Balanced (16GB RAM recommended)
*For: Modern laptops, Mac Mini, entry desktops*

```bash
ollama pull llama3.2         # 3B, great balance
ollama pull mistral          # 7B, strong reasoning
ollama pull gemma2:2b        # Google's small model
ollama pull phi3:medium      # Microsoft's mid-size
```

**Best for:** General use, good quality, reasonable speed.

### Tier 3: Powerful (32GB+ RAM or good GPU)
*For: Workstations, Mac Studio, gaming PCs*

```bash
ollama pull llama3.2:70b-q4  # Meta's flagship (quantized)
ollama pull mixtral          # 8x7B MoE, very capable
ollama pull codellama:34b    # Strong for coding
ollama pull deepseek-coder-v2:16b  # Excellent coder
```

**Best for:** Complex reasoning, coding, research.

### Tier 4: Maximum (64GB+ RAM, pro GPU)
*For: Servers, ML workstations*

```bash
ollama pull llama3.1:405b-q4  # Near-GPT4 level
ollama pull deepseek-coder-v2:236b  # World-class coding
ollama pull qwen2:72b         # Strong multilingual
```

**Best for:** Maximum capability, research, production.

## Specialized Models

### For Coding
```bash
ollama pull deepseek-coder-v2    # Best open-source coder
ollama pull codellama            # Meta's code model
ollama pull starcoder2           # Code completion specialist
```

### For Reasoning
```bash
ollama pull mistral              # Strong logical reasoning
ollama pull llama3.2             # Good general reasoning
ollama pull phi3                 # Punches above its weight
```

### For Long Context
```bash
ollama pull mistral-nemo         # 128k context
ollama pull llama3.1             # 128k context
ollama pull qwen2                # 32k context
```

### For Vision (describe images)
```bash
ollama pull llava                # Vision + language
ollama pull bakllava             # Better image understanding
ollama pull moondream            # Lightweight vision
```

## Hardware Tips

### Mac (Apple Silicon)
- M1/M2: Use 7B or smaller models
- M1/M2 Pro: Can run 13B comfortably
- M1/M2 Max: Can run 34B models
- M1/M2 Ultra: Can run 70B+ models

```bash
# Check your memory
sysctl hw.memsize | awk '{print $2/1024/1024/1024 " GB"}'
```

### GPU (NVIDIA)
- RTX 3060 (12GB): 7B-13B models
- RTX 3090 (24GB): 34B models
- RTX 4090 (24GB): 34B models faster
- A100 (40/80GB): 70B+ models

### CPU Only
Totally works! Just slower.
- 16GB RAM: Run 7B models
- 32GB RAM: Run 13B models
- 64GB RAM: Run 34B models

## Forever Mode Settings

For 24/7 operation, consider:

```python
ForeverConfig(
    # Longer intervals = less CPU usage
    background_interval_seconds=600,     # 10 min
    exploration_interval_seconds=7200,   # 2 hours
    
    # Sleep during off hours
    sleep_hours=(1, 6),  # 1AM-6AM
    
    # Don't hog resources
    max_cpu_percent=30,
)
```

## Cost Comparison

| Option | Monthly Cost | Forever Agent Cost |
|--------|--------------|-------------------|
| GPT-4 API | $50-500+ | N/A (metered) |
| Claude API | $50-500+ | N/A (metered) |
| Local Ollama | $0 | **$0** |
| Electricity (local) | ~$2-10 | ~$2-10 |

**Local = 50-100x cheaper for always-on operation.**

## Getting Started

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.2

# Test it
ollama run llama3.2 "Hello, are you ready to learn forever?"

# Start the forever agent
python -c "from src.forever import run_forever; run_forever()"
```

---

*The best model is the one that runs sustainably on your hardware.*
