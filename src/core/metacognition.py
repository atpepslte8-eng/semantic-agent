"""
Meta-Cognition - The agent thinks about its own thinking.

This is the WHY engine - not just doing things, but understanding
why they work or fail, and how to think better.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from datetime import datetime
import json


class ReasoningLevel(Enum):
    """Levels of reasoning depth - escalate when stuck."""
    
    QUICK = 0           # Pattern match from memory
    DELIBERATE = 1      # Step-by-step reasoning
    DEEP = 2            # Multi-angle analysis, why-chains
    DIAGNOSTIC = 3      # "I'm stuck. What am I missing?"
    SYNTHESIS = 4       # Gather from ALL knowledge sources


@dataclass
class Diagnosis:
    """Result of analyzing a failure."""
    failure_id: str
    surface_cause: str
    
    # The 5 WHYs
    why_chain: List[str]
    root_cause: str
    
    # What was missing
    missing_knowledge: List[str]
    wrong_assumptions: List[str]
    
    # Similar past failures
    similar_failures: List[str]
    
    # Path forward
    suggested_approach: str
    knowledge_to_acquire: List[str]
    
    # Confidence
    confidence: float = 0.0
    reasoning_trace: str = ""


@dataclass
class ThinkingState:
    """Current state of the agent's reasoning."""
    current_goal: Optional[str] = None
    sub_goals: List[str] = field(default_factory=list)
    
    # Reasoning state
    reasoning_level: ReasoningLevel = ReasoningLevel.QUICK
    attempts_at_current_level: int = 0
    max_attempts_before_escalate: int = 2
    
    # Stuck detection
    recent_failures: List[str] = field(default_factory=list)
    repeated_error_count: int = 0
    last_progress_at: datetime = field(default_factory=datetime.now)
    
    # Knowledge state
    known_unknowns: List[str] = field(default_factory=list)  # Things I know I don't know
    assumptions: List[str] = field(default_factory=list)      # Things I'm assuming


class MetaCognition:
    """
    The meta-cognitive layer - thinks about thinking.
    
    Responsibilities:
    - Analyze failures deeply (5 WHYs)
    - Detect when stuck
    - Escalate reasoning when needed
    - Track assumptions
    - Identify knowledge gaps
    """
    
    def __init__(self, memory, llm_fn: Callable[[str], str]):
        """
        Args:
            memory: SemanticMemory instance
            llm_fn: Function that takes a prompt and returns LLM response
        """
        self.memory = memory
        self.llm = llm_fn
        self.state = ThinkingState()
        
        # Track patterns
        self.failure_patterns: Dict[str, int] = {}  # pattern -> count
    
    def analyze_failure(self, failure_description: str, context: str = "") -> Diagnosis:
        """
        Deep analysis of a failure - find the ROOT cause.
        
        Uses the 5 WHYs technique, augmented with:
        - Memory of similar failures
        - Assumption checking
        - Knowledge gap identification
        """
        
        # Check if we've seen this pattern before
        similar_failures = self.memory.find_relevant_lessons(failure_description)
        similar_ids = [l.id for l in similar_failures[:3]]
        
        # Build the 5 WHYs
        why_chain = self._five_whys(failure_description, context)
        root_cause = why_chain[-1] if why_chain else failure_description
        
        # Identify what was missing
        missing = self._identify_missing_knowledge(failure_description, why_chain, context)
        assumptions = self._check_assumptions(failure_description, context)
        
        # Generate suggested approach
        approach = self._suggest_approach(
            failure_description, 
            root_cause, 
            similar_failures,
            missing
        )
        
        diagnosis = Diagnosis(
            failure_id=f"diag-{datetime.now().timestamp()}",
            surface_cause=failure_description,
            why_chain=why_chain,
            root_cause=root_cause,
            missing_knowledge=missing,
            wrong_assumptions=assumptions,
            similar_failures=similar_ids,
            suggested_approach=approach,
            knowledge_to_acquire=self._knowledge_to_acquire(missing, root_cause),
            confidence=0.7 if similar_failures else 0.5,
        )
        
        # Update failure pattern tracking
        self._track_failure_pattern(failure_description)
        
        return diagnosis
    
    def _five_whys(self, failure: str, context: str) -> List[str]:
        """Perform 5 WHYs analysis using LLM."""
        
        prompt = f"""Analyze this failure using the 5 WHYs technique.
For each WHY, dig deeper into the cause until you reach a root cause.

Failure: {failure}
Context: {context}

Respond with exactly 5 levels of WHY, each going deeper:

WHY 1: [surface cause]
WHY 2: [why did WHY 1 happen?]
WHY 3: [why did WHY 2 happen?]
WHY 4: [why did WHY 3 happen?]
WHY 5: [root cause - the deepest why]

Be specific and actionable. Don't be vague."""

        response = self.llm(prompt)
        
        # Parse the response
        whys = []
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("WHY"):
                # Extract the content after "WHY N:"
                parts = line.split(":", 1)
                if len(parts) > 1:
                    whys.append(parts[1].strip())
        
        return whys[:5]  # Ensure max 5
    
    def _identify_missing_knowledge(
        self, 
        failure: str, 
        why_chain: List[str],
        context: str
    ) -> List[str]:
        """Identify what knowledge was missing that led to failure."""
        
        prompt = f"""Given this failure and its root cause analysis, what knowledge was MISSING that would have prevented the failure?

Failure: {failure}
Why chain: {' -> '.join(why_chain)}
Context: {context}

List 2-4 specific pieces of knowledge that were missing:
1. 
2.
3.
4.

Be specific - not "more knowledge about X" but "the fact that X behaves Y when Z"."""

        response = self.llm(prompt)
        
        missing = []
        for line in response.split("\n"):
            line = line.strip()
            if line and line[0].isdigit() and "." in line:
                content = line.split(".", 1)[1].strip()
                if content:
                    missing.append(content)
        
        return missing[:4]
    
    def _check_assumptions(self, failure: str, context: str) -> List[str]:
        """Identify assumptions that were wrong."""
        
        prompt = f"""What assumptions were made (implicitly or explicitly) that turned out to be WRONG?

Failure: {failure}
Context: {context}

List assumptions that were incorrect:
1.
2.
3.

Format: "Assumed X, but actually Y"."""

        response = self.llm(prompt)
        
        assumptions = []
        for line in response.split("\n"):
            line = line.strip()
            if line and line[0].isdigit() and "." in line:
                content = line.split(".", 1)[1].strip()
                if content:
                    assumptions.append(content)
        
        return assumptions[:3]
    
    def _suggest_approach(
        self,
        failure: str,
        root_cause: str,
        similar_failures: List,
        missing_knowledge: List[str]
    ) -> str:
        """Suggest a better approach based on analysis."""
        
        similar_lessons = ""
        if similar_failures:
            similar_lessons = "\n".join([
                f"- Past lesson: {l.lesson_learned}" 
                for l in similar_failures[:3]
            ])
        
        prompt = f"""Based on this failure analysis, suggest a better approach.

Failure: {failure}
Root cause: {root_cause}
Missing knowledge: {', '.join(missing_knowledge)}
{f'Similar past lessons: {similar_lessons}' if similar_lessons else ''}

Provide a concise, actionable approach to prevent this failure:"""

        return self.llm(prompt).strip()
    
    def _knowledge_to_acquire(self, missing: List[str], root_cause: str) -> List[str]:
        """Identify what knowledge should be acquired."""
        
        knowledge = []
        
        for item in missing:
            knowledge.append(f"Learn: {item}")
        
        knowledge.append(f"Understand deeply: {root_cause}")
        
        return knowledge
    
    def _track_failure_pattern(self, failure: str):
        """Track failure patterns to detect repetition."""
        
        # Simple pattern extraction - could be more sophisticated
        pattern = failure.lower()[:50]  # First 50 chars as pattern key
        
        self.failure_patterns[pattern] = self.failure_patterns.get(pattern, 0) + 1
        self.state.recent_failures.append(failure)
        
        # Keep only recent failures
        if len(self.state.recent_failures) > 10:
            self.state.recent_failures = self.state.recent_failures[-10:]
        
        # Update repeated error count
        if self.failure_patterns[pattern] > 1:
            self.state.repeated_error_count += 1
    
    def detect_stuck(self) -> bool:
        """Detect if the agent is stuck (spinning wheels)."""
        
        # Check for repeated failures
        if self.state.repeated_error_count >= 2:
            return True
        
        # Check for no progress
        time_since_progress = (datetime.now() - self.state.last_progress_at).total_seconds()
        if time_since_progress > 300 and self.state.recent_failures:  # 5 minutes
            return True
        
        # Check for same errors repeating
        if len(self.state.recent_failures) >= 3:
            last_three = self.state.recent_failures[-3:]
            if len(set(last_three)) == 1:  # All same
                return True
        
        return False
    
    def escalate_reasoning(self) -> ReasoningLevel:
        """Escalate to deeper reasoning when stuck or struggling."""
        
        self.state.attempts_at_current_level += 1
        
        if self.state.attempts_at_current_level >= self.state.max_attempts_before_escalate:
            # Move to next level
            current = self.state.reasoning_level.value
            if current < ReasoningLevel.SYNTHESIS.value:
                self.state.reasoning_level = ReasoningLevel(current + 1)
                self.state.attempts_at_current_level = 0
        
        return self.state.reasoning_level
    
    def get_reasoning_prompt(self) -> str:
        """Get the appropriate reasoning prompt for current level."""
        
        level = self.state.reasoning_level
        
        prompts = {
            ReasoningLevel.QUICK: "",  # No special prompting
            
            ReasoningLevel.DELIBERATE: """
Think through this step by step:
1. What exactly is being asked?
2. What do I know that's relevant?
3. What's my approach?
4. Execute carefully.
5. Verify the result.
""",
            
            ReasoningLevel.DEEP: """
Deep analysis required. Consider:
- WHY is this the right approach? Are there alternatives?
- What assumptions am I making? Are they valid?
- What could go wrong?
- What would someone with more expertise do differently?
- Have I seen similar situations? What happened?

Think from multiple angles before proceeding.
""",
            
            ReasoningLevel.DIAGNOSTIC: """
I'm stuck. Time for diagnostic thinking:
- What have I tried? Why didn't it work?
- What am I missing? What don't I know that I should?
- What would I do differently if starting fresh?
- Is there a completely different approach?
- Who/what could I ask for help?

Don't repeat what didn't work. Find the real blocker.
""",
            
            ReasoningLevel.SYNTHESIS: """
Maximum effort required. Synthesize from ALL sources:
- What does my memory say about similar situations?
- What patterns apply here?
- What lessons from past failures are relevant?
- What blueprints could help?
- What's the fundamental nature of this problem?
- What would a genius-level expert do?

Take time. Think deeply. Find the insight.
"""
        }
        
        return prompts.get(level, "")
    
    def mark_progress(self):
        """Mark that progress was made (reset stuck detection)."""
        self.state.last_progress_at = datetime.now()
        self.state.repeated_error_count = 0
        self.state.recent_failures = []
        
        # Reset reasoning level on success
        self.state.reasoning_level = ReasoningLevel.QUICK
        self.state.attempts_at_current_level = 0
    
    def add_assumption(self, assumption: str):
        """Track an assumption being made."""
        self.state.assumptions.append(assumption)
    
    def add_known_unknown(self, unknown: str):
        """Track something we know we don't know."""
        self.state.known_unknowns.append(unknown)
    
    def reflect(self, action: str, result: str, success: bool) -> str:
        """
        Reflect on an action and its result.
        
        Returns reflection text that should be stored.
        """
        
        prompt = f"""Reflect on this action and result:

Action: {action}
Result: {result}
Success: {success}

Reflect:
1. WHY did this {'succeed' if success else 'fail'}?
2. What can be learned from this?
3. Would I do anything differently next time?
4. What knowledge should be extracted and remembered?

Be concise but insightful."""

        reflection = self.llm(prompt)
        
        if success:
            self.mark_progress()
        
        return reflection
    
    def identify_knowledge_gaps(self, goal: str) -> List[str]:
        """Identify knowledge gaps for achieving a goal."""
        
        prompt = f"""To achieve this goal, what knowledge might I be missing?

Goal: {goal}

What I need to know to succeed (but might not know):
1.
2.
3.
4.
5.

Be specific about knowledge gaps, not vague."""

        response = self.llm(prompt)
        
        gaps = []
        for line in response.split("\n"):
            line = line.strip()
            if line and line[0].isdigit() and "." in line:
                content = line.split(".", 1)[1].strip()
                if content:
                    gaps.append(content)
        
        self.state.known_unknowns.extend(gaps)
        return gaps
