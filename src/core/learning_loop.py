"""
Learning Loop - Every action is a learning opportunity.

The core cycle:
ACT → OBSERVE → REFLECT → EXTRACT → STORE → APPLY

This is what makes the agent actually LEARN, not just execute.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum
import json
import traceback

from .memory import (
    SemanticMemory, 
    Knowledge, 
    KnowledgeType, 
    Lesson, 
    Blueprint,
    Experience,
    generate_id
)
from .metacognition import MetaCognition, Diagnosis


@dataclass
class ActionSpec:
    """Specification for an action to perform."""
    action_type: str
    description: str
    intent: str                    # WHY are we doing this?
    expected_outcome: str          # What do we expect to happen?
    domain: List[str]              # What area is this in?
    
    # Optional
    parameters: Dict[str, Any] = field(default_factory=dict)
    prerequisites: List[str] = field(default_factory=list)


@dataclass
class ActionResult:
    """Result of executing an action."""
    success: bool
    output: Any
    error: Optional[str] = None
    
    # Observations
    actual_outcome: str = ""
    unexpected_observations: List[str] = field(default_factory=list)
    
    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime = field(default_factory=datetime.now)
    
    @property
    def duration_seconds(self) -> float:
        return (self.completed_at - self.started_at).total_seconds()


@dataclass
class LearningOutput:
    """What was learned from an action."""
    experience: Experience
    reflection: str
    
    # Extracted knowledge
    knowledge_items: List[Knowledge] = field(default_factory=list)
    lesson: Optional[Lesson] = None
    blueprint: Optional[Blueprint] = None
    
    # Meta
    reasoning_level_used: int = 0
    was_stuck: bool = False


class LearningLoop:
    """
    The core learning engine.
    
    Wraps every action in a learning cycle that:
    1. Prepares (recalls relevant knowledge)
    2. Executes (performs the action)
    3. Observes (captures what happened)
    4. Reflects (analyzes why)
    5. Extracts (pulls out learnings)
    6. Stores (updates memory)
    
    Over time, this makes the agent genuinely smarter.
    """
    
    def __init__(
        self,
        memory: SemanticMemory,
        meta: MetaCognition,
        llm_fn: Callable[[str], str],
        executor: Callable[[ActionSpec], ActionResult] = None
    ):
        """
        Args:
            memory: Semantic memory system
            meta: Meta-cognition system
            llm_fn: LLM function for reasoning
            executor: Function that actually performs actions
        """
        self.memory = memory
        self.meta = meta
        self.llm = llm_fn
        self.executor = executor
        
        # Stats
        self.total_actions = 0
        self.successful_actions = 0
        self.lessons_learned = 0
        self.blueprints_created = 0
    
    def execute(self, action: ActionSpec, context: str = "") -> LearningOutput:
        """
        Execute an action with full learning loop.
        
        This is the core method - every action goes through here.
        """
        
        self.total_actions += 1
        
        # ============ 1. PREPARE ============
        # Recall relevant knowledge before acting
        preparation = self._prepare(action, context)
        
        # Check if we're stuck and need to escalate reasoning
        if self.meta.detect_stuck():
            self.meta.escalate_reasoning()
        
        # Get appropriate reasoning guidance
        reasoning_prompt = self.meta.get_reasoning_prompt()
        
        # ============ 2. ACT ============
        result = self._act(action, preparation, reasoning_prompt)
        
        # ============ 3. OBSERVE ============
        observations = self._observe(action, result)
        
        # ============ 4. REFLECT ============
        reflection = self._reflect(action, result, observations, context)
        
        # ============ 5. EXTRACT ============
        extracted = self._extract(action, result, reflection, observations)
        
        # ============ 6. STORE ============
        self._store(extracted)
        
        # Update stats
        if result.success:
            self.successful_actions += 1
            self.meta.mark_progress()
        
        # Create experience record
        experience = Experience(
            id=generate_id(f"{action.description}-{datetime.now().isoformat()}"),
            timestamp=datetime.now(),
            action=action.description,
            intent=action.intent,
            result=result.actual_outcome or str(result.output),
            success=result.success,
            domain=action.domain,
            preceding_context=context[:500],  # Truncate
            reflection=reflection,
            extracted_knowledge=[k.id for k in extracted.get("knowledge", [])]
        )
        
        # Store experience
        self.memory.record_experience(experience)
        
        return LearningOutput(
            experience=experience,
            reflection=reflection,
            knowledge_items=extracted.get("knowledge", []),
            lesson=extracted.get("lesson"),
            blueprint=extracted.get("blueprint"),
            reasoning_level_used=self.meta.state.reasoning_level.value,
            was_stuck=self.meta.detect_stuck()
        )
    
    def _prepare(self, action: ActionSpec, context: str) -> Dict[str, Any]:
        """
        Prepare for action by recalling relevant knowledge.
        
        Returns a preparation package with:
        - Relevant memories
        - Applicable blueprints
        - Relevant lessons (especially from failures)
        """
        
        # Build query from action description and intent
        query = f"{action.description} {action.intent} {' '.join(action.domain)}"
        
        # Recall relevant knowledge
        relevant_knowledge = self.memory.recall(
            query=query,
            context=action.domain,
            top_k=5
        )
        
        # Find applicable blueprints
        blueprints = self.memory.find_applicable_blueprints(action.description)
        
        # Find relevant lessons (especially important!)
        lessons = self.memory.find_relevant_lessons(action.description)
        
        return {
            "knowledge": relevant_knowledge,
            "blueprints": blueprints[:2],  # Top 2 most relevant
            "lessons": lessons[:3],         # Top 3 relevant failures to avoid
            "warnings": [l.prevention for l in lessons[:3]]  # What to avoid
        }
    
    def _act(
        self, 
        action: ActionSpec, 
        preparation: Dict[str, Any],
        reasoning_prompt: str
    ) -> ActionResult:
        """Execute the action."""
        
        started_at = datetime.now()
        
        try:
            if self.executor:
                # Use provided executor
                result = self.executor(action)
            else:
                # Default: use LLM to simulate/reason about the action
                result = self._llm_execute(action, preparation, reasoning_prompt)
            
            result.started_at = started_at
            result.completed_at = datetime.now()
            return result
            
        except Exception as e:
            return ActionResult(
                success=False,
                output=None,
                error=str(e),
                actual_outcome=f"Exception: {str(e)}",
                unexpected_observations=[traceback.format_exc()],
                started_at=started_at,
                completed_at=datetime.now()
            )
    
    def _llm_execute(
        self, 
        action: ActionSpec, 
        preparation: Dict[str, Any],
        reasoning_prompt: str
    ) -> ActionResult:
        """Use LLM to execute/simulate an action."""
        
        # Build context from preparation
        context_parts = []
        
        if preparation["knowledge"]:
            context_parts.append("Relevant knowledge:")
            for k in preparation["knowledge"][:3]:
                context_parts.append(f"  - {k.content}")
        
        if preparation["blueprints"]:
            context_parts.append("\nApplicable blueprints:")
            for b in preparation["blueprints"]:
                context_parts.append(f"  - {b.name}: {b.key_insight}")
        
        if preparation["warnings"]:
            context_parts.append("\n⚠️ Lessons from past failures:")
            for w in preparation["warnings"]:
                context_parts.append(f"  - {w}")
        
        context_str = "\n".join(context_parts) if context_parts else ""
        
        prompt = f"""{reasoning_prompt}

{context_str}

ACTION: {action.description}
INTENT: {action.intent}
EXPECTED OUTCOME: {action.expected_outcome}

Execute this action thoughtfully. Report:
1. What you did
2. What happened (actual outcome)
3. Any unexpected observations
4. Success or failure"""

        response = self.llm(prompt)
        
        # Parse response (simplified)
        success = "success" in response.lower() and "fail" not in response.lower()
        
        return ActionResult(
            success=success,
            output=response,
            actual_outcome=response[:500],
            unexpected_observations=[]
        )
    
    def _observe(self, action: ActionSpec, result: ActionResult) -> Dict[str, Any]:
        """
        Carefully observe what happened.
        
        Compare expected vs actual outcome.
        Note anything unexpected.
        """
        
        observations = {
            "expected": action.expected_outcome,
            "actual": result.actual_outcome,
            "matched_expectation": False,
            "unexpected": result.unexpected_observations,
            "duration": result.duration_seconds
        }
        
        # Check if expectation matched
        prompt = f"""Did the actual outcome match the expected outcome?

Expected: {action.expected_outcome}
Actual: {result.actual_outcome}

Answer: YES or NO, then explain briefly."""

        response = self.llm(prompt)
        observations["matched_expectation"] = response.strip().upper().startswith("YES")
        observations["match_analysis"] = response
        
        return observations
    
    def _reflect(
        self, 
        action: ActionSpec, 
        result: ActionResult,
        observations: Dict[str, Any],
        context: str
    ) -> str:
        """
        Reflect on what happened and WHY.
        
        This is where real learning happens.
        """
        
        prompt = f"""Reflect deeply on this action and its result.

ACTION: {action.description}
INTENT: {action.intent}
SUCCESS: {result.success}

Expected: {action.expected_outcome}
Actual: {result.actual_outcome}
{'Error: ' + result.error if result.error else ''}

Matched expectations: {observations.get('matched_expectation', 'unknown')}

REFLECT:
1. WHY did this {'succeed' if result.success else 'fail'}?
2. What was the ROOT CAUSE of the outcome?
3. What assumptions were made? Were they correct?
4. What would I do differently?
5. What's the key insight/lesson here?

Be specific and dig deep. Don't accept surface explanations."""

        reflection = self.llm(prompt)
        
        # Also use meta-cognition for failures
        if not result.success:
            diagnosis = self.meta.analyze_failure(
                result.error or result.actual_outcome,
                context
            )
            reflection += f"\n\nDEEP DIAGNOSIS:\nRoot cause: {diagnosis.root_cause}\nMissing knowledge: {', '.join(diagnosis.missing_knowledge)}"
        
        return reflection
    
    def _extract(
        self, 
        action: ActionSpec, 
        result: ActionResult,
        reflection: str,
        observations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract learnings from the experience.
        
        Pull out:
        - Knowledge (facts, insights)
        - Lessons (from failures)
        - Blueprints (reusable patterns from successes)
        """
        
        extracted = {
            "knowledge": [],
            "lesson": None,
            "blueprint": None
        }
        
        # Always extract some knowledge
        knowledge_prompt = f"""What knowledge should be remembered from this experience?

Action: {action.description}
Result: {'Success' if result.success else 'Failure'}
Reflection: {reflection}

Extract 1-3 pieces of knowledge worth remembering:
1. [content] - WHY: [why this matters]
2. [content] - WHY: [why this matters]
3. [content] - WHY: [why this matters]"""

        knowledge_response = self.llm(knowledge_prompt)
        
        # Parse knowledge items
        for line in knowledge_response.split("\n"):
            if line.strip() and line[0].isdigit():
                parts = line.split(" - WHY: ")
                content = parts[0].split(".", 1)[-1].strip()
                why = parts[1].strip() if len(parts) > 1 else None
                
                if content:
                    k = Knowledge(
                        id=generate_id(content),
                        type=KnowledgeType.INSIGHT if result.success else KnowledgeType.FACT,
                        content=content,
                        why=why,
                        context=action.domain,
                        source=f"experience:{action.description[:50]}"
                    )
                    extracted["knowledge"].append(k)
        
        # Extract lesson from failure
        if not result.success:
            lesson = self._extract_lesson(action, result, reflection)
            if lesson:
                extracted["lesson"] = lesson
                self.lessons_learned += 1
        
        # Extract blueprint from significant success
        if result.success and len(reflection) > 200:  # Substantial reflection
            blueprint = self._extract_blueprint(action, result, reflection)
            if blueprint:
                extracted["blueprint"] = blueprint
                self.blueprints_created += 1
        
        return extracted
    
    def _extract_lesson(
        self, 
        action: ActionSpec, 
        result: ActionResult,
        reflection: str
    ) -> Optional[Lesson]:
        """Extract a lesson from a failure."""
        
        prompt = f"""Extract a formal LESSON from this failure.

Failure: {result.error or result.actual_outcome}
Action: {action.description}
Reflection: {reflection}

Provide:
SURFACE_CAUSE: [what went wrong at the surface]
WHY_CHAIN: [why 1] -> [why 2] -> [why 3] -> [why 4] -> [root why]
ROOT_CAUSE: [the deepest why]
LESSON: [what to remember]
PREVENTION: [how to prevent this in future]
TRIGGER: [when should this lesson be recalled]"""

        response = self.llm(prompt)
        
        # Parse response
        surface_cause = ""
        why_chain = []
        root_cause = ""
        lesson_text = ""
        prevention = ""
        trigger = ""
        
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("SURFACE_CAUSE:"):
                surface_cause = line.split(":", 1)[1].strip()
            elif line.startswith("WHY_CHAIN:"):
                chain_text = line.split(":", 1)[1].strip()
                why_chain = [w.strip() for w in chain_text.split("->")]
            elif line.startswith("ROOT_CAUSE:"):
                root_cause = line.split(":", 1)[1].strip()
            elif line.startswith("LESSON:"):
                lesson_text = line.split(":", 1)[1].strip()
            elif line.startswith("PREVENTION:"):
                prevention = line.split(":", 1)[1].strip()
            elif line.startswith("TRIGGER:"):
                trigger = line.split(":", 1)[1].strip()
        
        if lesson_text and root_cause:
            return Lesson(
                id=generate_id(f"lesson-{action.description}-{datetime.now().isoformat()}"),
                failure_description=result.error or result.actual_outcome,
                surface_cause=surface_cause,
                why_chain=why_chain,
                root_cause=root_cause,
                lesson_learned=lesson_text,
                prevention=prevention,
                domain=action.domain,
                trigger_pattern=trigger
            )
        
        return None
    
    def _extract_blueprint(
        self, 
        action: ActionSpec, 
        result: ActionResult,
        reflection: str
    ) -> Optional[Blueprint]:
        """Extract a reusable blueprint from a success."""
        
        prompt = f"""This action succeeded. Is it worth creating a reusable BLUEPRINT?

Action: {action.description}
Intent: {action.intent}
Outcome: {result.actual_outcome}
Reflection: {reflection}

If this is a pattern worth reusing, provide:
NAME: [short name for this pattern]
TRIGGER: [when to use this]
WHY_IT_WORKS: [why this approach succeeds]
KEY_INSIGHT: [the core realization]
STEPS: [step 1] | [step 2] | [step 3]
ANTI_PATTERNS: [what NOT to do]

If this is too specific/one-off, respond: NO_BLUEPRINT"""

        response = self.llm(prompt)
        
        if "NO_BLUEPRINT" in response:
            return None
        
        # Parse response
        name = ""
        trigger = ""
        why_it_works = ""
        key_insight = ""
        steps = []
        anti_patterns = []
        
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("NAME:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("TRIGGER:"):
                trigger = line.split(":", 1)[1].strip()
            elif line.startswith("WHY_IT_WORKS:"):
                why_it_works = line.split(":", 1)[1].strip()
            elif line.startswith("KEY_INSIGHT:"):
                key_insight = line.split(":", 1)[1].strip()
            elif line.startswith("STEPS:"):
                steps_text = line.split(":", 1)[1].strip()
                steps = [s.strip() for s in steps_text.split("|")]
            elif line.startswith("ANTI_PATTERNS:"):
                ap_text = line.split(":", 1)[1].strip()
                anti_patterns = [a.strip() for a in ap_text.split("|")]
        
        if name and key_insight:
            return Blueprint(
                id=generate_id(f"blueprint-{name}-{datetime.now().isoformat()}"),
                name=name,
                description=action.intent,
                trigger=trigger,
                context=action.domain,
                why_it_works=why_it_works,
                key_insight=key_insight,
                steps=steps,
                anti_patterns=anti_patterns,
                success_examples=[result.actual_outcome[:200]],
                failure_examples=[]
            )
        
        return None
    
    def _store(self, extracted: Dict[str, Any]):
        """Store extracted learnings in memory."""
        
        # Store knowledge
        for k in extracted.get("knowledge", []):
            self.memory.remember(k)
        
        # Store lesson
        if extracted.get("lesson"):
            self.memory.learn_lesson(extracted["lesson"])
        
        # Store blueprint
        if extracted.get("blueprint"):
            self.memory.add_blueprint(extracted["blueprint"])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            "total_actions": self.total_actions,
            "successful_actions": self.successful_actions,
            "success_rate": self.successful_actions / max(1, self.total_actions),
            "lessons_learned": self.lessons_learned,
            "blueprints_created": self.blueprints_created,
            "knowledge_items": len(self.memory.knowledge),
            "reasoning_level": self.meta.state.reasoning_level.name
        }
