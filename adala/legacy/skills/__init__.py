import warnings

warnings.warn(
    "The adala.skills module is deprecated and will be removed in a future version. "
    "Use adala.core.DataProcessor or adala.core.Classifier instead.",
    DeprecationWarning,
    stacklevel=2
)

from .skillset import SkillSet, LinearSkillSet, ParallelSkillSet
from .collection.classification import ClassificationSkill
from .collection.entity_extraction import EntityExtraction
from .collection.rag import RAGSkill
from .collection.ontology_creation import OntologyCreator, OntologyMerger
from .collection.label_studio import LabelStudioSkill
from ._base import Skill, TransformSkill, AnalysisSkill, SynthesisSkill
