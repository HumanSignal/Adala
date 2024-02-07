
from enum import Enum, unique


@unique
class EnvsType(str, Enum):
    STATIC = "StaticEnvironment"
    WEB_STATIC = "WebStaticEnvironment"
    

@unique
class RuntimesType(str, Enum):
    OPENAI_CHAT = "OpenAIChatRuntime"
    OPENAI_VISION = "OpenAIVisionRuntime"
    GUIDANCE = "GuidanceRuntime"
    LANG_CHAIN = "LangChainRuntime"


@unique
class SkillsType(str, Enum):
    QA = "QuestionAnswering"
    CLASSIFICATION = "ClassificationSkill"
    SUMMARY = "Summarization"
    GENERATION = "TextGeneration"
    TRANSLATION = "Translation"

    TRANSFORMATION = "Transformation"
    ANALYSIS = "Analysis"
    SYNTHESIS = "Synthesis"


@unique
class SkillsGroupType(str, Enum):
    LINEAR = "LinearSkillSet"
    PARALLEL = "ParallelSkillSet"


@unique
class SkillVersionType(str, Enum):
    ACTIVE = "Active"
    SYSTEM = "System"
    USER_DRAFT = "UserDraft"


@unique
class LearnRunState(str, Enum):
    INIT = "Initializing"
    RUNNING = "Running"
    FAILED = "Failed"
    DONE = "Done"


@unique
class ExecutionLogState(str, Enum):
    INIT = "Initializing"
    RUNNING = "Running"
    FAILED = "Failed"
    DONE = "Done"
