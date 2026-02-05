#!/usr/bin/env python3
"""
Semantic Agent CLI - Interact with a learning agent.

Usage:
    python cli.py [--brain BRAIN] [--storage PATH]
    
Examples:
    python cli.py                           # Use Ollama default
    python cli.py --brain ollama/llama3.2   # Local Ollama
    python cli.py --brain openai/gpt-4o     # OpenAI
    python cli.py --brain anthropic/claude-3-5-sonnet  # Anthropic
    python cli.py --brain groq/llama-3.2-90b  # Fast Groq
    python cli.py --brain http://localhost:8080/v1  # Custom endpoint
"""

import argparse
import os
import sys
from typing import List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agent import SemanticAgent
from brains import get_brain, list_brains


def create_embedding_function(brain, fallback: str = "local"):
    """Create an embedding function, preferring the brain's if available."""
    
    from brains.base import BrainCapability
    
    # Try to use brain's embedding
    if brain and brain.has_capability(BrainCapability.EMBEDDING):
        return brain.embed
    
    if fallback == "openai":
        def openai_embed(text: str) -> List[float]:
            import openai
            client = openai.OpenAI()
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        return openai_embed
    
    else:
        # Simple local embeddings
        def simple_embed(text: str) -> List[float]:
            """Simple hash-based embedding for demo."""
            import hashlib
            hash_bytes = hashlib.sha256(text.encode()).digest()
            return [b / 255.0 for b in hash_bytes][:32]
        
        return simple_embed


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Brain formats:
  ollama/llama3.2          Local Ollama model
  openai/gpt-4o            OpenAI API
  anthropic/claude-3-opus  Anthropic API
  groq/llama-3.2-90b       Groq (fast)
  deepseek/deepseek-chat   DeepSeek
  http://localhost:8080/v1 Any OpenAI-compatible endpoint

Environment variables:
  OPENAI_API_KEY       For OpenAI
  ANTHROPIC_API_KEY    For Anthropic
  GROQ_API_KEY         For Groq
  DEEPSEEK_API_KEY     For DeepSeek
  OPENROUTER_API_KEY   For OpenRouter
"""
    )
    parser.add_argument(
        "--brain", "-b",
        default="ollama/llama3.2",
        help="Brain to use (see formats below)"
    )
    parser.add_argument(
        "--storage", "-s",
        default="./agent_data",
        help="Storage path for agent memory"
    )
    parser.add_argument(
        "--name", "-n",
        default="Semantic",
        help="Agent name"
    )
    parser.add_argument(
        "--embeddings", "-e",
        default="auto",
        choices=["auto", "local", "openai"],
        help="Embedding provider (auto uses brain if capable)"
    )
    parser.add_argument(
        "--list-brains",
        action="store_true",
        help="List available brain providers"
    )
    # Legacy support
    parser.add_argument("--model", help=argparse.SUPPRESS)
    
    args = parser.parse_args()
    
    # Handle --list-brains
    if args.list_brains:
        brains = list_brains()
        print("\n🧠 Available Brain Providers\n")
        print("LOCAL (run on your hardware):")
        for name, desc in brains["local"].items():
            print(f"  {name:12} - {desc}")
        print("\nCLOUD APIs:")
        for name, desc in brains["cloud"].items():
            print(f"  {name:12} - {desc}")
        print()
        sys.exit(0)
    
    # Legacy --model support
    if args.model:
        args.brain = args.model
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    SEMANTIC AGENT                            ║
║                                                              ║
║  An agent that learns, understands WHY, and evolves.         ║
║  Pluggable brain: any LLM, local or cloud.                   ║
╚══════════════════════════════════════════════════════════════╝

Brain: {args.brain}
Storage: {args.storage}

Commands:
  /status   - Show agent status
  /diagnose - Deep diagnose a problem
  /goal     - Set a goal
  /reflect  - Reflect on session
  /compress - Compress memory
  /brain    - Show brain info
  /quit     - Exit

""")
    
    # Initialize brain
    print("Initializing brain...")
    try:
        brain = get_brain(args.brain)
        # Test connection
        print(f"  Connecting to {brain.name}...")
        brain.ping()
        print(f"✓ Brain connected: {brain.name}")
    except Exception as e:
        print(f"✗ Failed to connect to brain: {e}")
        if "ollama" in args.brain.lower():
            print("  Make sure Ollama is running: ollama serve")
        elif "api_key" in str(e).lower() or "api key" in str(e).lower():
            print("  Check your API key environment variable")
        sys.exit(1)
    
    # Create embedding function
    embed_fn = create_embedding_function(
        brain, 
        "local" if args.embeddings == "auto" else args.embeddings
    )
    print("✓ Embeddings ready")
    
    # Create agent
    print("Loading agent...")
    agent = SemanticAgent(
        name=args.name,
        storage_path=args.storage,
        brain=brain,
        embedding_fn=embed_fn
    )
    print(f"✓ Agent '{args.name}' ready\n")
    
    # Show initial status
    status = agent.get_status()
    print(f"Memory: {status['memory_stats']['knowledge']} knowledge items, "
          f"{status['memory_stats']['lessons']} lessons, "
          f"{status['memory_stats']['blueprints']} blueprints\n")
    
    # Main loop
    while True:
        try:
            user_input = input(f"\n{args.name}> ").strip()
            
            if not user_input:
                continue
            
            # Commands
            if user_input.startswith("/"):
                cmd = user_input.lower().split()[0]
                
                if cmd == "/quit" or cmd == "/exit":
                    print("\nReflecting on session before exit...")
                    reflection = agent.reflect_on_session()
                    print(f"\n{reflection[:500]}...")
                    print("\nGoodbye!")
                    break
                
                elif cmd == "/status":
                    status = agent.get_status()
                    print(f"""
Status:
  Name: {status['name']}
  Goal: {status['current_goal'] or 'None set'}
  Reasoning Level: {status['reasoning_level']}
  Stuck: {status['is_stuck']}
  
Memory:
  Knowledge: {status['memory_stats']['knowledge']}
  Lessons: {status['memory_stats']['lessons']}
  Blueprints: {status['memory_stats']['blueprints']}
  Experiences: {status['memory_stats']['experiences']}
  
Learning:
  Total Actions: {status['learning_stats']['total_actions']}
  Success Rate: {status['learning_stats']['success_rate']:.1%}
""")
                
                elif cmd == "/diagnose":
                    problem = " ".join(user_input.split()[1:])
                    if not problem:
                        problem = input("Describe the problem: ")
                    
                    print("\nAnalyzing (this may take a moment)...")
                    diagnosis = agent.diagnose(problem)
                    
                    print(f"""
DIAGNOSIS
=========
Surface Cause: {diagnosis['surface_cause']}

WHY Chain:
""")
                    for i, why in enumerate(diagnosis['why_chain'], 1):
                        print(f"  {i}. {why}")
                    
                    print(f"""
ROOT CAUSE: {diagnosis['root_cause']}

Missing Knowledge:
""")
                    for mk in diagnosis['missing_knowledge']:
                        print(f"  - {mk}")
                    
                    print(f"""
Suggested Approach:
{diagnosis['suggested_approach']}
""")
                
                elif cmd == "/goal":
                    goal = " ".join(user_input.split()[1:])
                    if not goal:
                        goal = input("What's your goal? ")
                    
                    result = agent.set_goal(goal)
                    print(f"\nGoal set: {goal}")
                    
                    if result['knowledge_gaps']:
                        print("\nKnowledge gaps identified:")
                        for gap in result['knowledge_gaps']:
                            print(f"  - {gap}")
                
                elif cmd == "/reflect":
                    print("\nReflecting on session...")
                    reflection = agent.reflect_on_session()
                    print(f"\n{reflection}")
                
                elif cmd == "/compress":
                    print("\nCompressing memory...")
                    result = agent.compress_memory()
                    print(f"Compressed {result['items_compressed']} items")
                    print(f"Total: {result['total_knowledge']} knowledge, "
                          f"{result['total_lessons']} lessons, "
                          f"{result['total_blueprints']} blueprints")
                
                elif cmd == "/brain":
                    from brains.base import BrainCapability
                    print(f"""
Brain Info:
  Name: {brain.name}
  Provider: {brain.config.provider}
  Model: {brain.config.model}
  
Capabilities:
  Text: ✓
  Streaming: {'✓' if brain.has_capability(BrainCapability.STREAMING) else '✗'}
  Embeddings: {'✓' if brain.has_capability(BrainCapability.EMBEDDING) else '✗'}
  Vision: {'✓' if brain.has_capability(BrainCapability.VISION) else '✗'}
""")
                
                elif cmd == "/switch":
                    new_brain_spec = " ".join(user_input.split()[1:])
                    if not new_brain_spec:
                        new_brain_spec = input("New brain (e.g., openai/gpt-4): ")
                    try:
                        brain = get_brain(new_brain_spec)
                        brain.ping()
                        agent.brain = brain
                        agent.llm = brain.think
                        print(f"✓ Switched to {brain.name}")
                    except Exception as e:
                        print(f"✗ Failed to switch: {e}")
                
                else:
                    print(f"Unknown command: {cmd}")
                
                continue
            
            # Normal conversation
            print("\nThinking...")
            response = agent.think(user_input)
            print(f"\n{response}")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Reflecting on session...")
            agent.reflect_on_session()
            print("Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
