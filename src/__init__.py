"""
Semantic Agent - An agent that learns, understands WHY, and treats knowledge as lifeblood.
"""

from agent import SemanticAgent
from core.memory import SemanticMemory, Knowledge, KnowledgeType, Lesson, Blueprint
from core.metacognition import MetaCognition, ReasoningLevel
from core.learning_loop import LearningLoop, ActionSpec, ActionResult

__version__ = "0.1.0"
__all__ = [
    "SemanticAgent",
    "SemanticMemory",
    "Knowledge",
    "KnowledgeType", 
    "Lesson",
    "Blueprint",
    "MetaCognition",
    "ReasoningLevel",
    "LearningLoop",
    "ActionSpec",
    "ActionResult"
]
