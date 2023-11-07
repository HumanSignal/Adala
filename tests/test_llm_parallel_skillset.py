import pandas as pd

from utils import patching, PatchedCalls

@patching(
    target_function=PatchedCalls.OPENAI_MODEL_LIST.value,
    data=[{'input': {}, 'output': {'data': [{'id': 'gpt-3.5-turbo-instruct'}]}}],
)
@patching(
    target_function=PatchedCalls.GUIDANCE.value,
    data=[
        {
            'input': {"text_": "Barack Obama was the 44th president of the United States."},
            'output': {"predictions": "Barack Obama"}
        },
        {
            'input': {"text_": "Apple's latest product, the iPhone 15, was released in September 2023."},
            'output': {"predictions": "Apple"}
        },
        {
            'input': {"text_": "Barack Obama was the 44th president of the United States."},
            'output': {"predictions": "United States"}
        },
        {
            'input': {"text_": "Apple's latest product, the iPhone 15, was released in September 2023."},
            'output': {"predictions": "September 2023"}
        },
        {
            'input': {"text_": "Barack Obama was the 44th president of the United States."},
            'output': {"predictions": "44th"}
        },
        {
            'input': {"text_": "Apple's latest product, the iPhone 15, was released in September 2023."},
            'output': {"predictions": "iPhone 15"}
        },
    ],
    strict=False
)
def test_llm_parallel_skillset():
    from adala.skills.skillset import ParallelSkillSet, LLMSkill
    from adala.datasets import DataFrameDataset, InternalDataFrame
    from adala.runtimes import OpenAIRuntime

    skillset = ParallelSkillSet(
        skills=[
            LLMSkill(name="skill_person", instructions="Extract person's name", input_data_field="text"),
            LLMSkill(name="skill_product", instructions="Extract product name", input_data_field="text"),
            LLMSkill(name="skill_date", instructions="Extract date", input_data_field="text"),
            LLMSkill(name="skill_location", instructions="Extract location", input_data_field="text"),
        ]
    )
    dataset = DataFrameDataset(df=InternalDataFrame([
        "Barack Obama was the 44th president of the United States.",
        "Apple's latest product, the iPhone 15, was released in September 2023.",
    ], columns=["text"]))
    predictions = skillset.apply(
        dataset=dataset,
        runtime=OpenAIRuntime(verbose=True),
    )

    pd.testing.assert_frame_equal(InternalDataFrame.from_records([
        {
            'text': 'Barack Obama was the 44th president of the United States.',
            'skill_person': 'Barack Obama',
            'skill_product': None,  # No product mentioned
            'skill_date': None,  # No date mentioned
            'skill_location': 'United States'
        },
        {
            'text': "Apple's latest product, the iPhone 15, was released in September 2023.",
            'skill_person': None,  # No person mentioned
            'skill_product': 'iPhone 15',
            'skill_date': 'September 2023',
            'skill_location': None  # No location mentioned
        },
    ]), predictions)