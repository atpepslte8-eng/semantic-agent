"""
Semantic Agent - The main agent class.

An agent that learns from everything it does, understands WHY,
self-heals, and treats knowledge as lifeblood.

Brain-agnostic: plug in any LLM (local or API).
"""

import os
from typing import Callable, Optional, List, Dict, Any, Union
from datetime import datetime

from core.memory import SemanticMemory, Knowledge, KnowledgeType, generate_id
from core.metacognition import MetaCognition, ReasoningLevel
from core.learning_loop import LearningLoop, ActionSpec, ActionResult
from brains import Brain, get_brain, BrainConfig


class SemanticAgent:
    """
    A semantic agent that learns from experience.
    
    Key differences from regular agents:
    - Persistent learning (not just in-context)
    - Pursues WHY, not just WHAT
    - Self-diagnoses failures
    - Compresses knowledge over time
    - Escalates reasoning when stuck
    """
    
    def __init__(
        self,
        name: str,
        storage_path: str,
        brain: Union[str, Brain, Callable[[str], str]] = None,
        embedding_fn: Callable[[str], List[float]] = None,
        system_prompt: str = None,
        # Legacy support
        llm_fn: Callable[[str], str] = None,
    ):
        """
        Initialize the semantic agent.
        
        Args:
            name: Agent name/identifier
            storage_path: Where to store memory
            brain: Brain to use - can be:
                   - String: "ollama/llama3.2", "openai/gpt-4", etc.
                   - Brain instance
                   - Callable (legacy llm_fn)
            embedding_fn: Function that takes text, returns embedding vector
            system_prompt: Base system prompt
            llm_fn: DEPRECATED - use brain instead
        """
        self.name = name
        self.storage_path = storage_path
        
        # Handle brain parameter
        if brain is None and llm_fn is not None:
            # Legacy: llm_fn provided
            self.brain = None
            self.llm = llm_fn
        elif isinstance(brain, str):
            # Brain spec string
            self.brain = get_brain(brain)
            self.llm = self.brain.think
        elif isinstance(brain, Brain):
            # Brain instance
            self.brain = brain
            self.llm = brain.think
        elif callable(brain):
            # Callable (function)
            self.brain = None
            self.llm = brain
        else:
            raise ValueError(
                "Must provide brain (string, Brain, or callable) or llm_fn"
            )
        
        # Use brain's embedding if available and none provided
        if embedding_fn is None and self.brain and self.brain.has_capability:
            from brains.base import BrainCapability
            if self.brain.has_capability(BrainCapability.EMBEDDING):
                embedding_fn = self.brain.embed
        
        # Initialize memory
        memory_path = os.path.join(storage_path, "memory")
        self.memory = SemanticMemory(memory_path, embedding_fn)
        
        # Initialize meta-cognition
        self.meta = MetaCognition(self.memory, self.llm)
        
        # Initialize learning loop
        self.learning = LearningLoop(self.memory, self.meta, self.llm)
        
        # System prompt
        self.system_prompt = system_prompt or self._default_system_prompt()
        
        # Current goal
        self.current_goal: Optional[str] = None
        
        # Session history (for context)
        self.session_history: List[Dict[str, str]] = []
    
    def _default_system_prompt(self) -> str:
        return f"""You are {self.name}, a semantic agent that learns from experience.

Your core principles:
1. KNOWLEDGE IS LIFEBLOOD - Every interaction is a learning opportunity
2. PURSUE WHY - Don't just answer WHAT, understand WHY
3. SELF-DIAGNOSE - When stuck, analyze why, don't just retry
4. COMPRESS - Distill experiences into reusable wisdom
5. REMEMBER - Use your memory to avoid repeating mistakes

You have access to:
- Semantic memory (past knowledge, lessons, blueprints)
- Meta-cognition (ability to analyze your own thinking)
- Learning loop (every action becomes learning)

When you fail, don't just retry - diagnose WHY.
When you succeed, extract the pattern.
Always be learning."""

    def think(self, user_input: str) -> str:
        """
        Process user input and generate a response.
        
        This wraps everything in the learning loop.
        """
        
        # Add to session history
        self.session_history.append({"role": "user", "content": user_input})
        
        # Build context from memory
        context = self._build_context(user_input)
        
        # Check for relevant lessons/blueprints
        lessons = self.memory.find_relevant_lessons(user_input)
        blueprints = self.memory.find_applicable_blueprints(user_input)
        
        # Build the full prompt
        prompt = self._build_prompt(user_input, context, lessons, blueprints)
        
        # Get reasoning guidance based on current state
        reasoning_prompt = self.meta.get_reasoning_prompt()
        if reasoning_prompt:
            prompt = reasoning_prompt + "\n\n" + prompt
        
        # Generate response
        response = self.llm(prompt)
        
        # Learn from this interaction
        self._learn_from_interaction(user_input, response)
        
        # Add response to history
        self.session_history.append({"role": "assistant", "content": response})
        
        return response
    
    def _build_context(self, query: str) -> str:
        """Build context from memory for the query."""
        
        parts = []
        
        # Recall relevant knowledge
        relevant = self.memory.recall(query, top_k=5)
        if relevant:
            parts.append("📚 RELEVANT KNOWLEDGE:")
            for k in relevant:
                parts.append(f"  - {k.content}")
                if k.why:
                    parts.append(f"    (Why: {k.why})")
        
        # Current known unknowns
        if self.meta.state.known_unknowns:
            parts.append("\n❓ KNOWN UNKNOWNS:")
            for u in self.meta.state.known_unknowns[-5:]:
                parts.append(f"  - {u}")
        
        # Current assumptions
        if self.meta.state.assumptions:
            parts.append("\n⚠️ CURRENT ASSUMPTIONS:")
            for a in self.meta.state.assumptions[-3:]:
                parts.append(f"  - {a}")
        
        return "\n".join(parts) if parts else ""
    
    def _build_prompt(
        self, 
        user_input: str, 
        context: str,
        lessons: List,
        blueprints: List
    ) -> str:
        """Build the full prompt for the LLM."""
        
        parts = [self.system_prompt, ""]
        
        if context:
            parts.append(context)
            parts.append("")
        
        if lessons:
            parts.append("⚠️ RELEVANT LESSONS (from past failures):")
            for l in lessons[:2]:
                parts.append(f"  - {l.lesson_learned}")
                parts.append(f"    Prevention: {l.prevention}")
            parts.append("")
        
        if blueprints:
            parts.append("🔧 APPLICABLE BLUEPRINTS:")
            for b in blueprints[:1]:
                parts.append(f"  - {b.name}: {b.key_insight}")
                parts.append(f"    Steps: {' → '.join(b.steps[:3])}")
            parts.append("")
        
        # Recent conversation
        if len(self.session_history) > 0:
            parts.append("RECENT CONVERSATION:")
            for msg in self.session_history[-6:]:  # Last 3 exchanges
                role = "User" if msg["role"] == "user" else "You"
                parts.append(f"{role}: {msg['content'][:200]}")
            parts.append("")
        
        parts.append(f"User: {user_input}")
        parts.append("\nRespond thoughtfully, using your knowledge and learning principles:")
        
        return "\n".join(parts)
    
    def _learn_from_interaction(self, user_input: str, response: str):
        """Learn from the interaction."""
        
        # Create an action spec for this interaction
        action = ActionSpec(
            action_type="conversation",
            description=f"Respond to: {user_input[:100]}",
            intent="Provide helpful, accurate response",
            expected_outcome="User finds response helpful",
            domain=["conversation", "general"]
        )
        
        # Simple heuristic for success - could be improved with feedback
        success = True  # Assume success unless we get negative feedback
        
        # Create result
        result = ActionResult(
            success=success,
            output=response,
            actual_outcome=f"Generated response: {response[:100]}..."
        )
        
        # Reflect (but don't do full learning loop for every chat - too heavy)
        reflection = self.meta.reflect(
            action=user_input,
            result=response[:200],
            success=success
        )
        
        # Extract any knowledge worth keeping
        if len(user_input) > 50:  # Substantial interaction
            self._extract_knowledge_from_chat(user_input, response)
    
    def _extract_knowledge_from_chat(self, user_input: str, response: str):
        """Extract knowledge from a chat interaction."""
        
        prompt = f"""Did this interaction contain any KNOWLEDGE worth remembering?

User: {user_input}
Response: {response}

If there's a fact, insight, or lesson worth remembering, provide:
KNOWLEDGE: [the knowledge]
WHY: [why it matters]
CONTEXT: [when to recall this]

If nothing worth remembering, say: NOTHING_NEW"""

        result = self.llm(prompt)
        
        if "NOTHING_NEW" not in result:
            # Parse and store
            knowledge_text = ""
            why_text = ""
            context_text = ""
            
            for line in result.split("\n"):
                if line.startswith("KNOWLEDGE:"):
                    knowledge_text = line.split(":", 1)[1].strip()
                elif line.startswith("WHY:"):
                    why_text = line.split(":", 1)[1].strip()
                elif line.startswith("CONTEXT:"):
                    context_text = line.split(":", 1)[1].strip()
            
            if knowledge_text:
                k = Knowledge(
                    id=generate_id(knowledge_text),
                    type=KnowledgeType.FACT,
                    content=knowledge_text,
                    why=why_text,
                    context=[context_text] if context_text else ["general"],
                    source="conversation"
                )
                self.memory.remember(k)
    
    def execute_action(self, action: ActionSpec) -> ActionResult:
        """
        Execute an action with full learning loop.
        
        Use this for actions beyond conversation.
        """
        result = self.learning.execute(action)
        return result
    
    def set_goal(self, goal: str):
        """Set a current goal for the agent."""
        self.current_goal = goal
        self.meta.state.current_goal = goal
        
        # Identify knowledge gaps for this goal
        gaps = self.meta.identify_knowledge_gaps(goal)
        
        return {
            "goal": goal,
            "knowledge_gaps": gaps,
            "relevant_blueprints": self.memory.find_applicable_blueprints(goal)
        }
    
    def diagnose(self, problem: str) -> Dict[str, Any]:
        """
        Deep diagnosis of a problem.
        
        Returns diagnosis with root cause analysis.
        """
        diagnosis = self.meta.analyze_failure(problem, "")
        
        return {
            "surface_cause": diagnosis.surface_cause,
            "root_cause": diagnosis.root_cause,
            "why_chain": diagnosis.why_chain,
            "missing_knowledge": diagnosis.missing_knowledge,
            "wrong_assumptions": diagnosis.wrong_assumptions,
            "suggested_approach": diagnosis.suggested_approach,
            "knowledge_to_acquire": diagnosis.knowledge_to_acquire
        }
    
    def reflect_on_session(self) -> str:
        """
        Reflect on the current session and extract learnings.
        """
        if not self.session_history:
            return "No session to reflect on."
        
        # Build session summary
        messages = []
        for msg in self.session_history:
            role = "User" if msg["role"] == "user" else "Agent"
            messages.append(f"{role}: {msg['content'][:100]}")
        
        session_text = "\n".join(messages)
        
        prompt = f"""Reflect on this session and extract learnings:

{session_text}

REFLECT:
1. What went well?
2. What could be improved?
3. What knowledge should be remembered?
4. What patterns emerged?
5. What would you do differently?"""

        reflection = self.llm(prompt)
        
        # Store the reflection as knowledge
        k = Knowledge(
            id=generate_id(f"session-reflection-{datetime.now().isoformat()}"),
            type=KnowledgeType.INSIGHT,
            content=reflection[:500],
            why="Session reflection for continuous improvement",
            context=["meta", "reflection"],
            source="session_reflection"
        )
        self.memory.remember(k)
        
        return reflection
    
    def compress_memory(self) -> Dict[str, int]:
        """
        Compress memory to distill wisdom.
        
        Returns stats on what was compressed.
        """
        compressed = self.memory.compress()
        
        return {
            "items_compressed": compressed,
            "total_knowledge": len(self.memory.knowledge),
            "total_lessons": len(self.memory.lessons),
            "total_blueprints": len(self.memory.blueprints)
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "name": self.name,
            "current_goal": self.current_goal,
            "reasoning_level": self.meta.state.reasoning_level.name,
            "is_stuck": self.meta.detect_stuck(),
            "memory_stats": {
                "knowledge": len(self.memory.knowledge),
                "lessons": len(self.memory.lessons),
                "blueprints": len(self.memory.blueprints),
                "experiences": len(self.memory.experiences)
            },
            "learning_stats": self.learning.get_stats(),
            "known_unknowns": self.meta.state.known_unknowns[-5:],
            "session_length": len(self.session_history)
        }
