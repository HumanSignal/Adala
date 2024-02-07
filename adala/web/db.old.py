
from enum import Enum

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# from main.core.config import get_app_settings

engine = create_engine(url="sqlite:///agents.db", echo=True, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator:
    """
    Generator dependency yield database connection.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()        

        
# db = SqliteDatabase('agents.db')
# db.connect()

class Database:    
    def __init__(self, db_path=':memory:'):
        self.db_path = db_path
        self.db = None

    def connect(self):
        # self.db = SqliteDatabase('agents.db')
        # self.db.connect()
        # db.connect()
        
        return db

    def close(self):
        if self.db:
            self.db.close()

            
class BaseModel(Model):
    class Meta:
        database = db

        
class AgentModel(BaseModel):
    id = AutoField()
    name = TextField()
    description = TextField(null=True)
    # environment = CharField()
    # skills = TextField()
    # memory = TextField(null=True)
    # runtimes = TextField()
    # default_runtime = OneToOneField('Runtime', backref='def_runtime')
    # default_teacher_runtime = OneToOneField('Runtime', backref='def_teached_runtime')
    # teacher_runtimes = TextField()
    

    def serialize(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            # 'environment': self.environment,
            # 'memory': self.memory,
            'runtimes': [ r.id for r in self.runtimes ],
            'default_runtime': self.default_runtime.name,
            'default_teacher_runtime': self.default_teacher_runtime.name
        }

    def serialize_skills(self):
        return [ s.serialize() for s in self.skills ]

    
class SkillModel(BaseModel):
    id = AutoField()
    agent = ForeignKeyField(AgentModel, backref='skills')
    
    class_name = CharField()
    name = CharField(unique=True)
    description = TextField(null=True)
    
    instructions = TextField()
    input_template = TextField()
    output_template = TextField()
    field_schema = TextField(null=True)
    verbose = BooleanField(default=False)

    def save_field_schema(self, field_schema):
        self.field_schema = json.dumps(field_schema)
        self.save()

    def load_field_schema(self):
        return json.loads(self.field_schema)

    def serialize(self):
        return {
            'id': self.id,
            'agent': self.agent.serialize(),
            'class_name': self.class_name,
            'name': self.name,
            'description': self.description,
            'instructions': self.instructions,
            'input_template': self.input_template,
            'output_template': self.output_template,
            'field_schema': self.load_field_schema(),
            'verbose': self.verbose
        }


class RuntimesEnum(Enum):
    OPENAI_CHAT = "OpenAIChatRuntime"
    OPENAI_VISION = "OpenAIVisionRuntime"
    GUIDANCE = "GuidanceRuntime"
    LANG_CHAIN = "LangChainRuntime"
    
    
class RuntimeModel(BaseModel):
    id = AutoField()

    name = CharField(unique=True)
    description = TextField(null=True)

    class_name = CharField(choices=RuntimesEnum)
    runtime_params = TextField(null=True)
    
    agents = ManyToManyField(AgentModel, backref="runtimes")
    agents_default = ForeignKeyField(AgentModel, null=True, backref="default_runtime")
    agents_default_teacher = ForeignKeyField(AgentModel, null=True, backref="default_teacher_runtime")
    
    def save_runtime_params(self, params):
        self.runtime_params = json.dumps(params)
        self.save()

    def load_runtime_params(self):
        return json.loads(self.runtime_params)
    
    def serialize(self):
        return {
            "id": self.id,
            "name": self.name,
            "class_name": self.class_name,
            "runtime_params": self.runtime_params
        }
    
