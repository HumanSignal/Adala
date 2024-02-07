

from .agent import AgentModel
from .executionlog import ExecutionLogModel
from .learnrun import LearnRunModel
from .runtime import  RuntimeModel
from .env import  EnvModel
from .skill import SkillModel
from .version import SkillVersionModel
from .associations import agent_runtime_assoc_table, agent_env_assoc_table


def init_models(engine): 
    for m in [AgentModel, agent_runtime_assoc_table, agent_env_assoc_table]:
        # m.metadata.drop_all(engine)
        m.metadata.create_all(engine)

