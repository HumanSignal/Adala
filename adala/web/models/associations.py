

from .base import Base
from sqlalchemy import Column, Integer, String, Text, Boolean, ForeignKey, Table, DateTime, Float

agent_runtime_assoc_table = Table('agent_runtime_association', Base.metadata,
    Column('agent_id', Integer, ForeignKey('agentmodel.id')),
    Column('runtime_id', Integer, ForeignKey('runtimemodel.id'))
)

agent_env_assoc_table = Table('agent_env_association', Base.metadata,
    Column('agent_id', Integer, ForeignKey('agentmodel.id')),
    Column('env_id', Integer, ForeignKey('envmodel.id'))
)
