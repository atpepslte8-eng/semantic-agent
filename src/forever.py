"""
Forever Mode - Continuous autonomous operation.

When running locally, there's no API cost. The agent can:
- Think as long as it needs
- Learn continuously in the background
- Explore and build knowledge proactively
- Run 24/7/365

This is the "forever agent" - always on, always learning, free to run.
"""

import os
import time
import threading
import queue
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from agent import SemanticAgent
from core.memory import Knowledge, KnowledgeType, generate_id
from core.learning_loop import ActionSpec


class ForeverMode(Enum):
    """Operating modes for the forever agent."""
    IDLE = "idle"               # Waiting for input
    ACTIVE = "active"           # Processing user request
    BACKGROUND = "background"   # Background learning/thinking
    EXPLORING = "exploring"     # Proactive knowledge building
    SLEEPING = "sleeping"       # Low-power mode (night, etc.)


@dataclass
class ForeverConfig:
    """Configuration for forever mode operation."""
    
    # Background processing
    background_thinking: bool = True       # Think in idle time
    background_interval_seconds: int = 300 # 5 minutes between background thoughts
    
    # Proactive exploration
    exploration_enabled: bool = True       # Explore topics proactively
    exploration_interval_seconds: int = 3600  # 1 hour between explorations
    
    # Memory maintenance
    auto_compress: bool = True             # Periodically compress memory
    compress_interval_hours: int = 24      # Daily compression
    
    # Resource management
    sleep_hours: tuple = (2, 6)           # Sleep during these hours (2AM-6AM)
    max_cpu_percent: int = 50              # Don't hog resources
    
    # Continuous learning
    learn_from_files: List[str] = field(default_factory=list)  # Watch these files
    learn_from_urls: List[str] = field(default_factory=list)   # Periodically fetch
    
    # Goals
    active_goals: List[str] = field(default_factory=list)  # What to work toward


@dataclass
class ThinkingTask:
    """A background thinking task."""
    id: str
    type: str           # "reflect", "explore", "learn", "compress"
    description: str
    priority: int = 5   # 1-10, higher = more important
    created_at: datetime = field(default_factory=datetime.now)


class ForeverAgent:
    """
    A Semantic Agent that runs forever.
    
    Key features:
    - No API costs (local model)
    - Continuous background learning
    - Proactive knowledge building
    - Self-maintaining memory
    - Goal-directed exploration
    
    The dream: an agent that gets smarter every day, for free.
    """
    
    def __init__(
        self,
        agent: SemanticAgent,
        config: ForeverConfig = None
    ):
        self.agent = agent
        self.config = config or ForeverConfig()
        
        # State
        self.mode = ForeverMode.IDLE
        self.running = False
        self.last_interaction = datetime.now()
        
        # Task queue for background work
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue()
        
        # Background thread
        self._background_thread: Optional[threading.Thread] = None
        
        # Stats
        self.stats = {
            "background_thoughts": 0,
            "explorations": 0,
            "compressions": 0,
            "total_runtime_hours": 0,
            "knowledge_gained": 0,
            "started_at": None
        }
        
        # Callbacks
        self.on_insight: Optional[Callable[[str], None]] = None  # Called when agent has an insight
        self.on_goal_progress: Optional[Callable[[str, float], None]] = None
    
    def start(self):
        """Start the forever agent."""
        if self.running:
            return
        
        self.running = True
        self.stats["started_at"] = datetime.now()
        
        # Start background thread
        self._background_thread = threading.Thread(
            target=self._background_loop,
            daemon=True
        )
        self._background_thread.start()
        
        print(f"🔄 Forever mode started - agent will run continuously")
        print(f"   Background thinking: {'✓' if self.config.background_thinking else '✗'}")
        print(f"   Exploration: {'✓' if self.config.exploration_enabled else '✗'}")
        print(f"   Auto-compress: {'✓' if self.config.auto_compress else '✗'}")
    
    def stop(self):
        """Stop the forever agent."""
        self.running = False
        if self._background_thread:
            self._background_thread.join(timeout=5)
        print("⏹️ Forever mode stopped")
    
    def interact(self, user_input: str) -> str:
        """Handle user interaction (foreground)."""
        self.mode = ForeverMode.ACTIVE
        self.last_interaction = datetime.now()
        
        response = self.agent.think(user_input)
        
        self.mode = ForeverMode.IDLE
        return response
    
    def add_goal(self, goal: str):
        """Add a goal for the agent to work toward."""
        self.config.active_goals.append(goal)
        
        # Queue exploration of this goal
        task = ThinkingTask(
            id=generate_id(f"goal-{goal}"),
            type="explore",
            description=f"Explore and understand: {goal}",
            priority=8  # High priority
        )
        self.task_queue.put((10 - task.priority, task))
    
    def _background_loop(self):
        """Main background processing loop."""
        last_background = datetime.now()
        last_exploration = datetime.now()
        last_compression = datetime.now()
        
        while self.running:
            try:
                now = datetime.now()
                
                # Check if we should sleep
                if self._should_sleep(now):
                    self.mode = ForeverMode.SLEEPING
                    time.sleep(60)  # Check every minute
                    continue
                
                # Don't background process if user is active
                if self._user_recently_active():
                    time.sleep(10)
                    continue
                
                # Process queued tasks first
                if not self.task_queue.empty():
                    self.mode = ForeverMode.BACKGROUND
                    _, task = self.task_queue.get_nowait()
                    self._process_task(task)
                    continue
                
                # Background thinking
                if self.config.background_thinking:
                    seconds_since = (now - last_background).total_seconds()
                    if seconds_since >= self.config.background_interval_seconds:
                        self.mode = ForeverMode.BACKGROUND
                        self._do_background_thinking()
                        last_background = now
                        continue
                
                # Exploration
                if self.config.exploration_enabled:
                    seconds_since = (now - last_exploration).total_seconds()
                    if seconds_since >= self.config.exploration_interval_seconds:
                        self.mode = ForeverMode.EXPLORING
                        self._do_exploration()
                        last_exploration = now
                        continue
                
                # Memory compression
                if self.config.auto_compress:
                    hours_since = (now - last_compression).total_seconds() / 3600
                    if hours_since >= self.config.compress_interval_hours:
                        self._do_compression()
                        last_compression = now
                        continue
                
                # Nothing to do, rest
                self.mode = ForeverMode.IDLE
                time.sleep(30)
                
            except Exception as e:
                print(f"Background error: {e}")
                time.sleep(60)
        
        # Update runtime stats
        if self.stats["started_at"]:
            runtime = datetime.now() - self.stats["started_at"]
            self.stats["total_runtime_hours"] = runtime.total_seconds() / 3600
    
    def _should_sleep(self, now: datetime) -> bool:
        """Check if it's sleep time."""
        hour = now.hour
        start, end = self.config.sleep_hours
        if start < end:
            return start <= hour < end
        else:  # Wraps midnight
            return hour >= start or hour < end
    
    def _user_recently_active(self) -> bool:
        """Check if user was active recently."""
        idle_time = (datetime.now() - self.last_interaction).total_seconds()
        return idle_time < 60  # Less than 1 minute ago
    
    def _process_task(self, task: ThinkingTask):
        """Process a queued background task."""
        if task.type == "explore":
            self._explore_topic(task.description)
        elif task.type == "reflect":
            self._do_reflection(task.description)
        elif task.type == "learn":
            self._learn_from_source(task.description)
        elif task.type == "compress":
            self._do_compression()
        
        self.stats["background_thoughts"] += 1
    
    def _do_background_thinking(self):
        """Do some background thinking - reflect, connect, synthesize."""
        
        # What should I think about?
        prompts = [
            "What patterns have I noticed in recent interactions that I should remember?",
            "What knowledge gaps do I have that I should fill?",
            "Are there connections between things I've learned that I haven't made explicit?",
            "What lessons from failures should I reinforce?",
            "What would make me more helpful?",
        ]
        
        import random
        prompt = random.choice(prompts)
        
        # Think about it
        reflection = self.agent.llm(f"""You are in background thinking mode. 
No user is waiting for a response. Take your time.

{prompt}

Think deeply and extract any insights worth remembering.""")
        
        # Extract and store any insights
        self._extract_insights(reflection, "background_thinking")
        
        self.stats["background_thoughts"] += 1
    
    def _do_exploration(self):
        """Proactively explore a topic to build knowledge."""
        
        # What to explore?
        if self.config.active_goals:
            # Work toward a goal
            import random
            goal = random.choice(self.config.active_goals)
            topic = goal
        else:
            # Explore something from memory gaps
            gaps = self.agent.meta.state.known_unknowns
            if gaps:
                import random
                topic = random.choice(gaps[-10:])  # Recent gaps
            else:
                # General exploration
                topic = "something useful that I should understand better"
        
        exploration = self.agent.llm(f"""You are in exploration mode. No user is waiting.

EXPLORE: {topic}

1. What are the key concepts I should understand?
2. Why does this matter? (The WHY)
3. How does this connect to things I already know?
4. What are common misconceptions or pitfalls?
5. What's a practical example?

Take your time. Build real understanding.""")
        
        # Store what was learned
        self._extract_insights(exploration, "exploration")
        
        self.stats["explorations"] += 1
    
    def _do_reflection(self, topic: str = None):
        """Reflect on experiences and extract wisdom."""
        
        prompt = "Reflect on recent experiences. What wisdom can be extracted?"
        if topic:
            prompt = f"Reflect deeply on: {topic}"
        
        reflection = self.agent.reflect_on_session() if not topic else self.agent.llm(prompt)
        
        self._extract_insights(reflection, "reflection")
    
    def _do_compression(self):
        """Compress and consolidate memory."""
        result = self.agent.compress_memory()
        self.stats["compressions"] += 1
        print(f"🗜️ Memory compressed: {result['items_compressed']} items consolidated")
    
    def _explore_topic(self, topic: str):
        """Deep dive into a specific topic."""
        
        # Multi-stage exploration
        stages = [
            f"What is {topic}? Explain the fundamentals.",
            f"Why does {topic} matter? What's the deeper significance?",
            f"What are the common problems/challenges with {topic}?",
            f"What are the best practices or patterns for {topic}?",
        ]
        
        all_insights = []
        for stage in stages:
            response = self.agent.llm(f"""Deep exploration mode.

{stage}

Provide substantive, useful information.""")
            all_insights.append(response)
        
        # Synthesize
        synthesis = self.agent.llm(f"""Synthesize these explorations into key knowledge:

{chr(10).join(all_insights)}

Extract:
1. Core facts worth remembering
2. Key insights (the WHY)
3. Practical patterns/blueprints""")
        
        self._extract_insights(synthesis, f"exploration:{topic}")
    
    def _learn_from_source(self, source: str):
        """Learn from a file or URL."""
        # TODO: Implement file/URL reading and learning
        pass
    
    def _extract_insights(self, text: str, source: str):
        """Extract and store insights from text."""
        
        extraction = self.agent.llm(f"""Extract knowledge worth remembering from this text:

{text}

For each piece of knowledge, provide:
KNOWLEDGE: [the fact or insight]
WHY: [why it matters]
TYPE: [fact/insight/pattern/lesson]

Only extract genuinely useful knowledge. Skip fluff.""")
        
        # Parse and store
        current_knowledge = None
        current_why = None
        current_type = None
        
        for line in extraction.split("\n"):
            line = line.strip()
            if line.startswith("KNOWLEDGE:"):
                if current_knowledge:
                    self._store_knowledge(current_knowledge, current_why, current_type, source)
                current_knowledge = line.split(":", 1)[1].strip()
                current_why = None
                current_type = None
            elif line.startswith("WHY:"):
                current_why = line.split(":", 1)[1].strip()
            elif line.startswith("TYPE:"):
                current_type = line.split(":", 1)[1].strip().lower()
        
        if current_knowledge:
            self._store_knowledge(current_knowledge, current_why, current_type, source)
    
    def _store_knowledge(self, content: str, why: str, ktype: str, source: str):
        """Store a piece of knowledge."""
        
        type_map = {
            "fact": KnowledgeType.FACT,
            "insight": KnowledgeType.INSIGHT,
            "pattern": KnowledgeType.PATTERN,
            "lesson": KnowledgeType.LESSON,
        }
        
        k = Knowledge(
            id=generate_id(content),
            type=type_map.get(ktype, KnowledgeType.FACT),
            content=content,
            why=why,
            context=["forever_mode", source],
            source=source
        )
        
        self.agent.memory.remember(k)
        self.stats["knowledge_gained"] += 1
        
        # Callback if there's an interesting insight
        if self.on_insight and ktype == "insight":
            self.on_insight(content)
    
    def get_status(self) -> Dict[str, Any]:
        """Get forever agent status."""
        
        runtime = None
        if self.stats["started_at"]:
            runtime = datetime.now() - self.stats["started_at"]
        
        return {
            "mode": self.mode.value,
            "running": self.running,
            "runtime": str(runtime) if runtime else None,
            "stats": self.stats,
            "config": {
                "background_thinking": self.config.background_thinking,
                "exploration": self.config.exploration_enabled,
                "auto_compress": self.config.auto_compress,
                "sleep_hours": self.config.sleep_hours,
            },
            "active_goals": self.config.active_goals,
            "queued_tasks": self.task_queue.qsize(),
            "agent_status": self.agent.get_status()
        }


def run_forever(
    brain: str = "ollama/llama3.2",
    storage: str = "./forever_agent",
    name: str = "Forever",
    goals: List[str] = None
):
    """
    Run an agent forever.
    
    Example:
        run_forever(
            brain="ollama/llama3.2",
            goals=["understand distributed systems", "master Python"]
        )
    """
    from brains import get_brain
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    FOREVER AGENT                             ║
║                                                              ║
║  Running locally = free forever                              ║
║  Always learning, always improving                           ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Initialize
    brain_instance = get_brain(brain)
    
    agent = SemanticAgent(
        name=name,
        storage_path=storage,
        brain=brain_instance
    )
    
    config = ForeverConfig(
        active_goals=goals or []
    )
    
    forever = ForeverAgent(agent, config)
    
    # Insight callback
    def on_insight(insight: str):
        print(f"\n💡 Insight: {insight[:100]}...")
    
    forever.on_insight = on_insight
    
    # Start forever mode
    forever.start()
    
    print("\nType your messages. The agent learns even when idle.")
    print("Commands: /status, /goals, /stop\n")
    
    try:
        while forever.running:
            try:
                user_input = input(f"\n{name}> ").strip()
                
                if not user_input:
                    continue
                
                if user_input == "/stop":
                    forever.stop()
                    break
                
                elif user_input == "/status":
                    status = forever.get_status()
                    print(f"""
Mode: {status['mode']}
Runtime: {status['runtime']}
Background thoughts: {status['stats']['background_thoughts']}
Explorations: {status['stats']['explorations']}
Knowledge gained: {status['stats']['knowledge_gained']}
Active goals: {status['active_goals']}
""")
                
                elif user_input.startswith("/goal"):
                    goal = user_input[5:].strip()
                    if goal:
                        forever.add_goal(goal)
                        print(f"✓ Goal added: {goal}")
                    else:
                        print(f"Goals: {forever.config.active_goals}")
                
                else:
                    response = forever.interact(user_input)
                    print(f"\n{response}")
                    
            except EOFError:
                break
                
    except KeyboardInterrupt:
        print("\n")
    
    forever.stop()
    
    # Final stats
    status = forever.get_status()
    print(f"""
Final Stats:
  Runtime: {status['runtime']}
  Background thoughts: {status['stats']['background_thoughts']}
  Explorations: {status['stats']['explorations']}
  Knowledge gained: {status['stats']['knowledge_gained']}
  
Memory: {status['agent_status']['memory_stats']['knowledge']} knowledge items
""")


if __name__ == "__main__":
    run_forever()
