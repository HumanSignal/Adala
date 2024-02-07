
import json
import pandas as pd

from pathlib import Path
from adala.agents import Agent

import adala.runtimes
import adala.skills
import adala.environments
import adala.skills.skillset

from .settings import UPLOAD_PATH

def create_adala_instance(agent):
    """ """
        ## load runtimes
    print(agent.runtimes)
    runtimes = {}

    # this should be an attribute on the model and configurable through the UI.
    default_runtime = ""
    
    for rt in agent.runtimes:
        class_obj = getattr(adala.runtimes, rt.rt_class_name)
        print(rt.runtime_params)
        print(rt.runtime_params)
        params = {}
        if rt.runtime_params is not None:
            params = json.loads(rt.runtime_params)
        
        instance = class_obj(**params)
        runtimes[rt.name] = instance

        default_runtime = rt.name


    skills = {}
    skills_map = {}
    
    for sk in agent.skills:
        kwargs = {}
        if sk.skill_params:
            kwargs = json.loads(sk.skill_params)
            
        field_schema = None
        if sk.field_schema:
            field_schema = json.loads(sk.field_schema)
            
        class_obj = getattr(adala.skills, sk.sk_class_name)

        print(kwargs)
        print(sk)
        
        instance = class_obj(name=sk.name,
                             # description=sk.description, 
                             instructions=sk.instructions,
                             input_template=sk.input_template,
                             output_template=sk.output_template,
                             field_schema=field_schema,
                             **kwargs)

        skills_map[sk.name] = sk
        skills[sk.name] = instance

    print(skills)
    
    skills_group_class = getattr(adala.skills.skillset, agent.skills_group_class_name)

    print(skills_group_class)
    print(agent.skills_group_class_name)
    
    sk_group_obj = skills_group_class(skills=skills)
    
    print(runtimes)

    # ENVs
    
    filename = Path(UPLOAD_PATH) / agent.envs[0].local_filename
    train_df = pd.read_csv(filename)

    env = agent.envs[0]
    env_class = getattr(adala.environments, env.env_class_name)
    env_obj = env_class(df=train_df)

    
    # train_df = pd.DataFrame([
    #     ["It was the negative first impressions, and then it started working.", "Positive"],
    #     ["Not loud enough and doesn't turn on like it should.", "Negative"],
    #     ["I don't know what to say.", "Neutral"],
    #     ["Manager was rude, but the most important that mic shows very flat frequency response.", "Positive"],
    #     ["The phone doesn't seem to accept anything except CBR mp3s.", "Negative"],
    #     ["I tried it before, I bought this device for my son.", "Neutral"],
    # ], columns=["text", "sentiment"])

    # # Test dataset
    # test_df = pd.DataFrame([
    #     "All three broke within two months of use.",
    #     "The device worked for a long time, can't say anything bad.",
    #     "Just a random line of text."
    # ], columns=["text"])

    adala_agent = Agent(
        # connect to a dataset
        environment=env_obj,
        
        # define a skill
        skills=sk_group_obj,
        
        # define all the different runtimes your skills may use
        runtimes = runtimes,
        
        default_runtime=default_runtime,
        
        # NOTE! If you have access to GPT-4, you can uncomment the lines bellow for better results
        #     default_teacher_runtime='openai-gpt4',
        #     teacher_runtimes = {
        #       'openai-gpt4': OpenAIRuntime(model='gpt-4')
        #     }
    )

    return adala_agent, skills_map, runtimes, env_obj
    # print(agent)
    # print(agent.skills)
