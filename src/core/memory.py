"""
Semantic Memory - The knowledge lifeblood of the agent.

Not just text storage - structured, connected, searchable knowledge
that grows and compresses over time.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import json
import hashlib


class KnowledgeType(Enum):
    FACT = "fact"           # Something known to be true
    PATTERN = "pattern"     # A recurring structure
    LESSON = "lesson"       # Learning from failure
    BLUEPRINT = "blueprint" # Reusable solution template
    INSIGHT = "insight"     # Deep understanding / WHY
    EXPERIENCE = "experience"  # Raw experience record


@dataclass
class Knowledge:
    """A single piece of knowledge."""
    id: str
    type: KnowledgeType
    content: str
    
    # Semantic richness
    why: Optional[str] = None          # Why is this true/important?
    context: List[str] = field(default_factory=list)  # When does this apply?
    connections: List[str] = field(default_factory=list)  # Related knowledge IDs
    
    # Source tracking
    source: Optional[str] = None       # Where did this come from?
    confidence: float = 1.0            # How sure are we?
    
    # Temporal
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    use_count: int = 0
    
    # Embedding (populated by memory system)
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "why": self.why,
            "context": self.context,
            "connections": self.connections,
            "source": self.source,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "use_count": self.use_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Knowledge":
        return cls(
            id=data["id"],
            type=KnowledgeType(data["type"]),
            content=data["content"],
            why=data.get("why"),
            context=data.get("context", []),
            connections=data.get("connections", []),
            source=data.get("source"),
            confidence=data.get("confidence", 1.0),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None,
            use_count=data.get("use_count", 0),
        )


@dataclass
class Lesson:
    """Learning from a failure - the 5 WHYs crystallized."""
    id: str
    failure_description: str
    
    # The why chain - digging to root cause
    surface_cause: str
    why_chain: List[str]  # Each level of WHY
    root_cause: str       # The deepest WHY
    
    # The learning
    lesson_learned: str
    prevention: str       # How to prevent next time
    
    # Context
    domain: List[str]     # What areas does this apply to?
    trigger_pattern: str  # When should I recall this?
    
    # Meta
    created_at: datetime = field(default_factory=datetime.now)
    times_applied: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "failure_description": self.failure_description,
            "surface_cause": self.surface_cause,
            "why_chain": self.why_chain,
            "root_cause": self.root_cause,
            "lesson_learned": self.lesson_learned,
            "prevention": self.prevention,
            "domain": self.domain,
            "trigger_pattern": self.trigger_pattern,
            "created_at": self.created_at.isoformat(),
            "times_applied": self.times_applied,
        }


@dataclass 
class Blueprint:
    """A reusable solution pattern - compressed wisdom."""
    id: str
    name: str
    description: str
    
    # When to use
    trigger: str              # What situation triggers this?
    context: List[str]        # What domain/area?
    
    # The understanding
    why_it_works: str         # WHY does this approach work?
    key_insight: str          # The core realization
    
    # The approach
    steps: List[str]          # How to apply
    anti_patterns: List[str]  # What NOT to do
    
    # Evidence
    success_examples: List[str]  # Times this worked
    failure_examples: List[str]  # Times it didn't (edge cases)
    
    # Meta
    created_at: datetime = field(default_factory=datetime.now)
    times_used: int = 0
    success_rate: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "trigger": self.trigger,
            "context": self.context,
            "why_it_works": self.why_it_works,
            "key_insight": self.key_insight,
            "steps": self.steps,
            "anti_patterns": self.anti_patterns,
            "success_examples": self.success_examples,
            "failure_examples": self.failure_examples,
            "created_at": self.created_at.isoformat(),
            "times_used": self.times_used,
            "success_rate": self.success_rate,
        }


@dataclass
class Experience:
    """A raw experience - action + result + context."""
    id: str
    timestamp: datetime
    
    # What happened
    action: str
    intent: str               # What was I trying to do?
    result: str
    success: bool
    
    # Context
    domain: List[str]
    preceding_context: str    # What led to this?
    
    # Reflection (filled in by learning loop)
    reflection: Optional[str] = None
    extracted_knowledge: List[str] = field(default_factory=list)  # Knowledge IDs
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "intent": self.intent,
            "result": self.result,
            "success": self.success,
            "domain": self.domain,
            "preceding_context": self.preceding_context,
            "reflection": self.reflection,
            "extracted_knowledge": self.extracted_knowledge,
        }


class SemanticMemory:
    """
    The agent's semantic memory system.
    
    Not just storage - this is living knowledge that:
    - Connects related concepts
    - Compresses redundant information  
    - Surfaces relevant context
    - Grows and evolves
    """
    
    def __init__(self, storage_path: str, embedding_fn=None):
        self.storage_path = storage_path
        self.embedding_fn = embedding_fn  # Function to generate embeddings
        
        # Core storage
        self.knowledge: Dict[str, Knowledge] = {}
        self.lessons: Dict[str, Lesson] = {}
        self.blueprints: Dict[str, Blueprint] = {}
        self.experiences: Dict[str, Experience] = {}
        
        # Indices for fast lookup
        self.by_type: Dict[KnowledgeType, List[str]] = {t: [] for t in KnowledgeType}
        self.by_context: Dict[str, List[str]] = {}  # context tag -> knowledge IDs
        
        # Load existing memory
        self._load()
    
    def remember(self, knowledge: Knowledge) -> str:
        """Store new knowledge, connecting it to existing knowledge."""
        
        # Generate embedding if we have the function
        if self.embedding_fn and not knowledge.embedding:
            knowledge.embedding = self.embedding_fn(knowledge.content)
        
        # Find and create connections to related knowledge
        if knowledge.embedding:
            related = self._find_similar(knowledge.embedding, top_k=5)
            knowledge.connections.extend([k.id for k in related if k.id != knowledge.id])
        
        # Check for redundancy - maybe we already know this?
        existing = self._find_redundant(knowledge)
        if existing:
            # Merge rather than duplicate
            self._merge_knowledge(existing, knowledge)
            return existing.id
        
        # Store
        self.knowledge[knowledge.id] = knowledge
        self.by_type[knowledge.type].append(knowledge.id)
        
        for ctx in knowledge.context:
            if ctx not in self.by_context:
                self.by_context[ctx] = []
            self.by_context[ctx].append(knowledge.id)
        
        self._save()
        return knowledge.id
    
    def learn_lesson(self, lesson: Lesson) -> str:
        """Store a lesson from failure."""
        self.lessons[lesson.id] = lesson
        
        # Also create a knowledge entry for searchability
        knowledge = Knowledge(
            id=f"knowledge-{lesson.id}",
            type=KnowledgeType.LESSON,
            content=lesson.lesson_learned,
            why=lesson.root_cause,
            context=lesson.domain,
            source=f"lesson:{lesson.id}",
        )
        self.remember(knowledge)
        
        self._save()
        return lesson.id
    
    def add_blueprint(self, blueprint: Blueprint) -> str:
        """Store a reusable blueprint."""
        self.blueprints[blueprint.id] = blueprint
        
        # Also create a knowledge entry
        knowledge = Knowledge(
            id=f"knowledge-{blueprint.id}",
            type=KnowledgeType.BLUEPRINT,
            content=f"{blueprint.name}: {blueprint.description}",
            why=blueprint.why_it_works,
            context=blueprint.context,
            source=f"blueprint:{blueprint.id}",
        )
        self.remember(knowledge)
        
        self._save()
        return blueprint.id
    
    def record_experience(self, experience: Experience) -> str:
        """Store a raw experience for later reflection."""
        self.experiences[experience.id] = experience
        self._save()
        return experience.id
    
    def recall(self, query: str, context: List[str] = None, top_k: int = 10) -> List[Knowledge]:
        """
        Recall relevant knowledge for a query.
        
        Uses:
        - Semantic similarity (embeddings)
        - Context matching
        - Recency and use frequency
        """
        results = []
        
        # Semantic search if we have embeddings
        if self.embedding_fn:
            query_embedding = self.embedding_fn(query)
            results = self._find_similar(query_embedding, top_k=top_k * 2)
        
        # Context boost
        if context:
            for ctx in context:
                if ctx in self.by_context:
                    for kid in self.by_context[ctx]:
                        k = self.knowledge.get(kid)
                        if k and k not in results:
                            results.append(k)
        
        # Update usage stats
        for k in results[:top_k]:
            k.last_used = datetime.now()
            k.use_count += 1
        
        self._save()
        return results[:top_k]
    
    def find_relevant_lessons(self, situation: str) -> List[Lesson]:
        """Find lessons that might apply to current situation."""
        relevant = []
        
        if self.embedding_fn:
            situation_embedding = self.embedding_fn(situation)
            
            for lesson in self.lessons.values():
                # Check trigger pattern similarity
                trigger_embedding = self.embedding_fn(lesson.trigger_pattern)
                similarity = self._cosine_similarity(situation_embedding, trigger_embedding)
                
                if similarity > 0.7:  # Threshold
                    relevant.append((similarity, lesson))
        
        # Sort by relevance
        relevant.sort(key=lambda x: x[0], reverse=True)
        return [l for _, l in relevant]
    
    def find_applicable_blueprints(self, situation: str) -> List[Blueprint]:
        """Find blueprints that might apply."""
        applicable = []
        
        if self.embedding_fn:
            situation_embedding = self.embedding_fn(situation)
            
            for blueprint in self.blueprints.values():
                trigger_embedding = self.embedding_fn(blueprint.trigger)
                similarity = self._cosine_similarity(situation_embedding, trigger_embedding)
                
                if similarity > 0.7:
                    applicable.append((similarity, blueprint))
        
        applicable.sort(key=lambda x: x[0], reverse=True)
        return [b for _, b in applicable]
    
    def compress(self) -> int:
        """
        Compress memory by:
        - Merging redundant knowledge
        - Creating blueprints from repeated patterns
        - Archiving rarely-used knowledge
        
        Returns number of items compressed.
        """
        compressed = 0
        
        # Find clusters of similar knowledge
        clusters = self._cluster_similar_knowledge()
        
        for cluster in clusters:
            if len(cluster) > 3:  # Multiple similar items
                # Create a compressed representation
                merged = self._merge_cluster(cluster)
                compressed += len(cluster) - 1
        
        # Archive old, unused knowledge
        threshold = datetime.now().timestamp() - (30 * 24 * 60 * 60)  # 30 days
        for k in list(self.knowledge.values()):
            if k.use_count == 0 and k.created_at.timestamp() < threshold:
                self._archive(k)
                compressed += 1
        
        self._save()
        return compressed
    
    def _find_similar(self, embedding: List[float], top_k: int = 5) -> List[Knowledge]:
        """Find knowledge similar to the given embedding."""
        similarities = []
        
        for k in self.knowledge.values():
            if k.embedding:
                sim = self._cosine_similarity(embedding, k.embedding)
                similarities.append((sim, k))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [k for _, k in similarities[:top_k]]
    
    def _find_redundant(self, knowledge: Knowledge) -> Optional[Knowledge]:
        """Check if we already have essentially the same knowledge."""
        if not knowledge.embedding:
            return None
            
        similar = self._find_similar(knowledge.embedding, top_k=1)
        if similar and self._cosine_similarity(knowledge.embedding, similar[0].embedding) > 0.95:
            return similar[0]
        return None
    
    def _merge_knowledge(self, existing: Knowledge, new: Knowledge):
        """Merge new knowledge into existing."""
        # Combine contexts
        for ctx in new.context:
            if ctx not in existing.context:
                existing.context.append(ctx)
        
        # Update confidence (reinforcement)
        existing.confidence = min(1.0, existing.confidence + 0.1)
        
        # Add to connections
        existing.connections.extend(new.connections)
        existing.connections = list(set(existing.connections))
    
    def _merge_cluster(self, cluster: List[Knowledge]) -> Knowledge:
        """Merge a cluster of similar knowledge into one."""
        # Use the most used/confident as base
        base = max(cluster, key=lambda k: k.use_count * k.confidence)
        
        for k in cluster:
            if k.id != base.id:
                self._merge_knowledge(base, k)
                del self.knowledge[k.id]
        
        return base
    
    def _cluster_similar_knowledge(self) -> List[List[Knowledge]]:
        """Cluster similar knowledge items."""
        # Simple clustering based on embedding similarity
        clusters = []
        used = set()
        
        for k in self.knowledge.values():
            if k.id in used or not k.embedding:
                continue
            
            cluster = [k]
            used.add(k.id)
            
            for other in self.knowledge.values():
                if other.id in used or not other.embedding:
                    continue
                
                if self._cosine_similarity(k.embedding, other.embedding) > 0.85:
                    cluster.append(other)
                    used.add(other.id)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    def _archive(self, knowledge: Knowledge):
        """Archive rarely-used knowledge (not delete, just move to cold storage)."""
        # In a real implementation, move to separate archive store
        del self.knowledge[knowledge.id]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot / (norm_a * norm_b)
    
    def _load(self):
        """Load memory from disk."""
        import os
        
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path, exist_ok=True)
            return
        
        # Load knowledge
        knowledge_file = os.path.join(self.storage_path, "knowledge.jsonl")
        if os.path.exists(knowledge_file):
            with open(knowledge_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    k = Knowledge.from_dict(data)
                    self.knowledge[k.id] = k
                    self.by_type[k.type].append(k.id)
                    for ctx in k.context:
                        if ctx not in self.by_context:
                            self.by_context[ctx] = []
                        self.by_context[ctx].append(k.id)
        
        # Load lessons
        lessons_file = os.path.join(self.storage_path, "lessons.jsonl")
        if os.path.exists(lessons_file):
            with open(lessons_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    self.lessons[data["id"]] = Lesson(**{
                        **data,
                        "created_at": datetime.fromisoformat(data["created_at"])
                    })
        
        # Load blueprints
        blueprints_file = os.path.join(self.storage_path, "blueprints.jsonl")
        if os.path.exists(blueprints_file):
            with open(blueprints_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    self.blueprints[data["id"]] = Blueprint(**{
                        **data,
                        "created_at": datetime.fromisoformat(data["created_at"])
                    })
    
    def _save(self):
        """Persist memory to disk."""
        import os
        
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Save knowledge
        with open(os.path.join(self.storage_path, "knowledge.jsonl"), "w") as f:
            for k in self.knowledge.values():
                f.write(json.dumps(k.to_dict()) + "\n")
        
        # Save lessons
        with open(os.path.join(self.storage_path, "lessons.jsonl"), "w") as f:
            for l in self.lessons.values():
                f.write(json.dumps(l.to_dict()) + "\n")
        
        # Save blueprints  
        with open(os.path.join(self.storage_path, "blueprints.jsonl"), "w") as f:
            for b in self.blueprints.values():
                f.write(json.dumps(b.to_dict()) + "\n")


def generate_id(content: str) -> str:
    """Generate a deterministic ID from content."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]
