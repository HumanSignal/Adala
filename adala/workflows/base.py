from pydantic import BaseModel


class AgentGraph(BaseModel):
    """
    Base class for agent graphs
    """
    pass


class Workflow(BaseModel):
    """
    Base class for workflows
    """
    agents: AgentGraph
