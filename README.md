# Semantic Agent 🧠

**An AI agent that actually learns.**

Not just a chatbot - a semantic agent that:
- **Learns** from everything it does
- **Understands WHY**, not just WHAT
- **Self-diagnoses** failures
- **Compresses** knowledge over time
- **Escalates** reasoning when stuck
- Treats **knowledge as lifeblood**

## The Problem with Current Agents

| Current Agents | Semantic Agent |
|----------------|----------------|
| Forgets between sessions | Remembers and builds knowledge |
| Answers WHAT | Pursues WHY |
| Retries blindly | Diagnoses failures |
| Reactive | Goal-driven |
| Accumulates text | Compresses to wisdom |
| No self-awareness | Meta-cognitive |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama (or use OpenAI/Anthropic)
ollama pull llama3.2

# Run the agent
python cli.py
```

## Usage

### CLI Mode

```bash
python cli.py --model ollama/llama3.2 --storage ./my-agent
```

Commands:
- `/status` - Show agent status
- `/diagnose <problem>` - Deep root-cause analysis
- `/goal <goal>` - Set a learning goal
- `/reflect` - Reflect on session
- `/compress` - Compress memory
- `/quit` - Exit

### Programmatic

```python
from src import SemanticAgent

# Create LLM function
def my_llm(prompt: str) -> str:
    # Your LLM call here
    return response

# Create agent
agent = SemanticAgent(
    name="Learner",
    storage_path="./agent_data",
    llm_fn=my_llm,
    embedding_fn=my_embedding_fn  # Optional
)

# Conversation
response = agent.think("How do I fix this bug?")

# Diagnose a problem
diagnosis = agent.diagnose("My code keeps timing out")
print(diagnosis['root_cause'])
print(diagnosis['suggested_approach'])

# Set a goal
agent.set_goal("Understand distributed systems")

# Get status
status = agent.get_status()
print(f"Knowledge: {status['memory_stats']['knowledge']}")
print(f"Lessons learned: {status['memory_stats']['lessons']}")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     SEMANTIC CORE                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │  Knowledge  │ │  Blueprints │ │   Lessons   │               │
│  │    Graph    │ │  (patterns) │ │  (failures) │               │
│  └─────────────┘ └─────────────┘ └─────────────┘               │
├─────────────────────────────────────────────────────────────────┤
│                     META-COGNITION LAYER                        │
│  WHY-Analyzer │ Self-Diagnosis │ Pattern-Finder │ Compressor   │
├─────────────────────────────────────────────────────────────────┤
│                     LEARNING LOOP                               │
│        ACT → OBSERVE → REFLECT → EXTRACT → STORE → APPLY       │
├─────────────────────────────────────────────────────────────────┤
│                     REASONING ESCALATION                        │
│   Quick → Deliberate → Deep → Diagnostic → Synthesis           │
├─────────────────────────────────────────────────────────────────┤
│                     LLM BACKBONE                                │
│           (Ollama / OpenAI / Anthropic - swappable)             │
└─────────────────────────────────────────────────────────────────┘
```

## Key Concepts

### Learning Loop

Every action goes through:

1. **ACT** - Perform the action
2. **OBSERVE** - Watch what happens
3. **REFLECT** - Analyze WHY
4. **EXTRACT** - Pull out patterns/lessons
5. **STORE** - Update memory
6. **APPLY** - Use in future

### Knowledge Types

- **Facts** - Things known to be true
- **Patterns** - Recurring structures
- **Lessons** - Learnings from failures
- **Blueprints** - Reusable solution templates
- **Insights** - Deep understanding (WHY)

### The 5 WHYs

When something fails, don't just retry. Ask WHY five times:

```
Failure: Request timed out
WHY 1: Server didn't respond
WHY 2: Server was overloaded
WHY 3: Too many concurrent requests
WHY 4: No rate limiting implemented
WHY 5: Didn't anticipate scale → ROOT CAUSE
```

### Reasoning Escalation

When stuck, automatically escalate:

- **Level 0: Quick** - Pattern match
- **Level 1: Deliberate** - Step-by-step
- **Level 2: Deep** - Multi-angle analysis
- **Level 3: Diagnostic** - "What am I missing?"
- **Level 4: Synthesis** - Pull from ALL knowledge

## Model Support

- **Ollama** (local): `ollama/llama3.2`, `ollama/mistral`
- **OpenAI**: `openai/gpt-4`, `openai/gpt-3.5-turbo`
- **Anthropic**: `anthropic/claude-3-opus`
- Any OpenAI-compatible API

## Philosophy

> "The goal is not to build a better chatbot. It's to build something that actually learns."

Knowledge is lifeblood. Every interaction is learning. Understanding WHY matters more than knowing WHAT. When stuck, diagnose, don't retry. Compress wisdom, don't accumulate text.

## License

MIT
