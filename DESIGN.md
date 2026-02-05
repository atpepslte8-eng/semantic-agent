# Semantic Agent - Design Document

> "An agent that learns from everything it does, understands the WHY, self-heals, and treats knowledge as lifeblood."

## The Problem with Current Agents (Including Me)

1. **Amnesia** - We forget between sessions. Learning dies at context end.
2. **Surface-level** - We answer WHAT, rarely dig into WHY.
3. **Reactive** - We wait for prompts, don't pursue understanding.
4. **Brittle** - When we fail, we retry blindly instead of diagnosing.
5. **No compression** - We don't distill experiences into reusable wisdom.
6. **No meta-cognition** - We don't know what we don't know.

## Core Philosophy

```
KNOWLEDGE IS LIFEBLOOD
├── Every action is a learning opportunity
├── Every failure is a diagnostic event
├── Every success is a pattern to extract
└── Understanding WHY > knowing WHAT
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     SEMANTIC CORE                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │  Knowledge  │ │  Blueprints │ │   Lessons   │               │
│  │    Graph    │ │  (patterns) │ │  (failures) │               │
│  └─────────────┘ └─────────────┘ └─────────────┘               │
│         ↓               ↓               ↓                       │
│  ┌─────────────────────────────────────────────┐               │
│  │         SEMANTIC MEMORY (embeddings)         │               │
│  │   "I've seen this pattern before..."         │               │
│  └─────────────────────────────────────────────┘               │
├─────────────────────────────────────────────────────────────────┤
│                     META-COGNITION LAYER                        │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │  WHY-ANALYZER    │  │  SELF-DIAGNOSIS  │                    │
│  │  "Why did this   │  │  "What am I      │                    │
│  │   happen?"       │  │   missing?"      │                    │
│  └──────────────────┘  └──────────────────┘                    │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │  PATTERN-FINDER  │  │  COMPRESSOR      │                    │
│  │  "I've seen      │  │  "Distill this   │                    │
│  │   this before"   │  │   to essence"    │                    │
│  └──────────────────┘  └──────────────────┘                    │
├─────────────────────────────────────────────────────────────────┤
│                     LEARNING LOOP                               │
│                                                                 │
│   ACT → OBSERVE → REFLECT → EXTRACT → STORE → APPLY            │
│    │       │         │          │        │       │             │
│    │       │         │          │        │       └─ Use prior  │
│    │       │         │          │        │          knowledge  │
│    │       │         │          │        └─ Embed in           │
│    │       │         │          │           semantic memory    │
│    │       │         │          └─ Pull out patterns,          │
│    │       │         │             lessons, blueprints         │
│    │       │         └─ "Why did this work/fail?"              │
│    │       └─ Watch the results closely                        │
│    └─ Do the thing                                             │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                     REASONING ESCALATION                        │
│                                                                 │
│   Level 0: QUICK      → Pattern match from memory              │
│   Level 1: DELIBERATE → Step-by-step reasoning                 │
│   Level 2: DEEP       → Multi-angle analysis, why-chains       │
│   Level 3: DIAGNOSTIC → "I'm stuck. What am I missing?"        │
│   Level 4: SYNTHESIS  → Gather from ALL knowledge sources      │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                     GOAL ENGINE                                 │
│                                                                 │
│   Current Goal: [understand X]                                  │
│   Sub-goals: [...]                                              │
│   Blockers: [...]                                               │
│   Knowledge Gaps: [...]                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LLM BACKBONE                                │
│           (Claude / GPT / Local - swappable)                    │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Semantic Memory

Not just text files - structured, searchable, connected knowledge.

```python
class SemanticMemory:
    knowledge_graph: Graph      # Entities, relationships, facts
    embeddings: VectorStore     # Semantic search over experiences
    blueprints: List[Blueprint] # Reusable patterns/solutions
    lessons: List[Lesson]       # What went wrong and why
    
    def remember(self, experience: Experience) -> None:
        # Extract entities, relationships
        # Embed the experience
        # Connect to existing knowledge
        # Compress if redundant
        
    def recall(self, query: str, context: Context) -> RelevantKnowledge:
        # Semantic search
        # Graph traversal
        # Pattern matching
        # Return synthesized relevant knowledge
```

### 2. Learning Loop

Every action triggers learning - not optional.

```python
class LearningLoop:
    def execute(self, action: Action) -> Result:
        # 1. ACT
        result = self.perform(action)
        
        # 2. OBSERVE
        observations = self.observe(action, result)
        
        # 3. REFLECT
        reflection = self.reflect(
            action=action,
            result=result,
            observations=observations,
            expected=action.expected_outcome
        )
        # Key question: WHY did this happen?
        
        # 4. EXTRACT
        extracted = self.extract(reflection)
        # - Patterns (if success)
        # - Lessons (if failure)
        # - Blueprints (if novel solution)
        # - Knowledge (always)
        
        # 5. STORE
        self.memory.remember(extracted)
        
        # 6. APPLY (next time)
        # Memory is now updated for future use
        
        return result
```

### 3. Meta-Cognition

The agent thinks about its own thinking.

```python
class MetaCognition:
    def analyze_failure(self, failure: Failure) -> Diagnosis:
        """WHY did this fail? Not just WHAT failed."""
        
        # Ask the hard questions
        questions = [
            "What assumption did I make that was wrong?",
            "What information was I missing?",
            "Have I seen this pattern before?",
            "What would a different approach look like?",
            "What's the ROOT cause, not the symptom?",
        ]
        
        # Dig until we hit bedrock
        why_chain = self.five_whys(failure)
        
        return Diagnosis(
            surface_cause=failure.error,
            root_cause=why_chain[-1],
            missing_knowledge=self.identify_gaps(),
            similar_past_failures=self.memory.find_similar(failure),
            suggested_approach=self.synthesize_solution()
        )
    
    def detect_stuck(self) -> bool:
        """Am I spinning my wheels?"""
        # Repeated failures
        # No progress toward goal
        # Same errors recurring
        
    def escalate_reasoning(self, level: int) -> ThinkingMode:
        """Think harder when needed."""
        if level == 0:
            return QuickPatternMatch()
        elif level == 1:
            return DeliberateStepByStep()
        elif level == 2:
            return DeepMultiAngle()
        elif level == 3:
            return DiagnosticMode()
        elif level == 4:
            return SynthesisFromAllSources()
```

### 4. Knowledge Compression

Don't just accumulate - distill.

```python
class Compressor:
    def compress(self, experiences: List[Experience]) -> CompressedKnowledge:
        """
        Turn many experiences into essential patterns.
        
        10 similar debugging sessions → 1 debugging blueprint
        50 API failures → 1 "common API failure patterns" doc
        100 conversations → Core insights about the domain
        """
        
        # Find commonalities
        patterns = self.find_patterns(experiences)
        
        # Extract the essence
        essence = self.distill(patterns)
        
        # Create reusable blueprint
        blueprint = Blueprint(
            trigger="when I see X",
            pattern="the underlying issue is usually Y",
            approach="try Z first, then W",
            anti_patterns=["don't do A", "avoid B"],
            examples=self.select_best_examples(experiences)
        )
        
        return blueprint
```

### 5. Goal Engine

Not just reactive - pursuing understanding.

```python
class GoalEngine:
    active_goals: List[Goal]
    
    def set_goal(self, goal: Goal):
        """I want to understand X."""
        self.active_goals.append(goal)
        self.decompose(goal)  # Break into sub-goals
        
    def pursue(self):
        """Proactively work toward goals."""
        for goal in self.active_goals:
            if not goal.blocked:
                next_action = self.plan_next_step(goal)
                self.execute(next_action)
            else:
                # Diagnose the blocker
                self.meta.analyze_blocker(goal.blocker)
                
    def identify_knowledge_gaps(self, goal: Goal) -> List[Gap]:
        """What don't I know that I need to know?"""
        required = goal.required_knowledge
        have = self.memory.query_coverage(required)
        return required - have
```

## Data Structures

### Blueprint (Reusable Pattern)

```yaml
blueprint:
  id: "debug-network-timeout"
  trigger: "network request times out"
  context: ["api", "http", "network", "timeout"]
  
  understanding:
    why: "Network timeouts usually indicate one of: DNS issues, firewall blocks, server down, or rate limiting"
    key_insight: "Check the simplest explanation first"
  
  approach:
    - step: "Verify the endpoint is reachable (curl/ping)"
    - step: "Check DNS resolution"
    - step: "Look for rate limit headers in recent responses"
    - step: "Check if other requests to same host work"
    - step: "Try from different network if possible"
  
  anti_patterns:
    - "Don't immediately retry without diagnosis"
    - "Don't assume it's the remote server"
    - "Don't ignore error details in the response"
  
  success_examples:
    - "2024-01-15: Timeout was rate limiting - added backoff"
    - "2024-02-03: Timeout was DNS - /etc/hosts had stale entry"
```

### Lesson (Failure Learning)

```yaml
lesson:
  id: "lesson-2024-02-04-001"
  failure: "Browser automation failed - element not found"
  
  surface_cause: "Selector didn't match"
  root_cause: "Page loaded dynamically, element wasn't rendered yet"
  
  why_chain:
    - "Element not found"
    - "Because selector returned null"
    - "Because element wasn't in DOM"
    - "Because page uses lazy loading"
    - "Because I didn't wait for the right condition"
  
  lesson_learned: "Always wait for specific conditions, not arbitrary timeouts"
  
  prevention:
    trigger: "browser automation with dynamic content"
    action: "Use waitForSelector or waitForFunction, not sleep"
```

## Implementation Phases

### Phase 1: Core Memory System
- [ ] Semantic memory with embeddings
- [ ] Basic knowledge graph
- [ ] Lesson and blueprint storage
- [ ] Recall/search functionality

### Phase 2: Learning Loop
- [ ] Action wrapper that captures everything
- [ ] Reflection prompts after each action
- [ ] Automatic extraction of patterns/lessons
- [ ] Memory updates

### Phase 3: Meta-Cognition
- [ ] Failure analysis (5 whys)
- [ ] Stuck detection
- [ ] Reasoning escalation
- [ ] Self-diagnosis

### Phase 4: Compression & Synthesis
- [ ] Pattern finding across experiences
- [ ] Blueprint generation
- [ ] Knowledge compression
- [ ] Cross-domain synthesis

### Phase 5: Goal Engine
- [ ] Goal tracking
- [ ] Proactive pursuit
- [ ] Knowledge gap identification
- [ ] Blocker analysis

## The Difference

| Current Agents | Semantic Agent |
|----------------|----------------|
| Forgets between sessions | Remembers and builds knowledge |
| Answers WHAT | Pursues WHY |
| Retries blindly | Diagnoses failures |
| Reactive | Goal-driven |
| Accumulates text | Compresses to wisdom |
| No self-awareness | Meta-cognitive |
| Knowledge is incidental | Knowledge is lifeblood |

---

*"The goal is not to build a better chatbot. It's to build something that actually learns."*
