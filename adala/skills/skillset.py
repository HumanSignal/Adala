from pydantic import BaseModel


class SkillSet(BaseModel):
    """
    Collection of interdependent skills
    Skill set aims at achieving a particular goal,
    therefore it splits the goal path into the skills that are needed to be acquired first.
    Agents can evolve skills in parallel (e.g. for self-consistency)
     or in sequence (e.g. for complex problem decomposition and causal reasoning)
    Most generic task can involve graph decomposition
    """
    pass


class LinearSkillSet(SkillSet):
    """
    Linear skill set is a sequence of skills that need to be acquired in order to achieve a goal
    """
    pass


class ParallelSkillSet(SkillSet):
    """
    Parallel skill set is a set of skills that can be acquired in parallel to achieve a goal
    """
    pass
