import pandas as pd

from utils import patching, PatchedCalls


@patching(
    target_function=PatchedCalls.OPENAI_MODEL_LIST.value,
    data=[{'input': {}, 'output': {'data': [{'id': 'gpt-3.5-turbo-instruct'}]}}],
)
@patching(
    target_function=PatchedCalls.GUIDANCE.value,
    data=[
        # Responses for the first text entry
        {
            'input': {"text_": "Apple's latest product, the iPhone 15, was released in September 2023."},
            'output': {"skill_person": ""}  # No person mentioned
        },
        {
            'input': {"text_": "Barack Obama was the 44th president of the United States."},
            'output': {"skill_person": "Barack Obama"}
        },
        {
            'input': {"text_": "Apple's latest product, the iPhone 15, was released in September 2023."},
            'output': {"skill_product": "iPhone 15"}
        },
        {
            'input': {"text_": "Barack Obama was the 44th president of the United States."},
            'output': {"skill_product": ""}  # No product mentioned
        },
        {
            'input': {"text_": "Apple's latest product, the iPhone 15, was released in September 2023."},
            'output': {"skill_date": "September 2023"}
        },
        {
            'input': {"text_": "Barack Obama was the 44th president of the United States."},
            'output': {"skill_date": ""}  # No date mentioned
        },
        {
            'input': {"text_": "Apple's latest product, the iPhone 15, was released in September 2023."},
            'output': {"skill_location": ""}  # No location mentioned
        },
        {
            'input': {"text_": "Barack Obama was the 44th president of the United States."},
            'output': {"skill_location": "United States"}
        }
    ],
    strict=False
)
def test_llm_parallel_skillset():
    from adala.skills import ParallelSkillSet, TransformSkill
    from adala.datasets import DataFrameDataset, InternalDataFrame
    from adala.runtimes import GuidanceRuntime

    skillset = ParallelSkillSet(
        skills=[
            TransformSkill(
                name="skill_person",
                instructions="Extract person's name",
                input_template="Text: {text}",
                output_template="Person: {skill_person}",
            ),
            TransformSkill(
                name="skill_product",
                instructions="Extract product name",
                input_template="Text: {text}",
                output_template="Product: {skill_product}",
            ),
            TransformSkill(
                name="skill_date",
                instructions="Extract date",
                input_template="Text: {text}",
                output_template="Date: {skill_date}",
            ),
            TransformSkill(
                name="skill_location",
                instructions="Extract location",
                input_template="Text: {text}",
                output_template="Location: {skill_location}",
            ),
        ]
    )
    df = InternalDataFrame([
        "Apple's latest product, the iPhone 15, was released in September 2023.",
        "Barack Obama was the 44th president of the United States.",
    ], columns=["text"])
    predictions = skillset.apply(
        input=df,
        runtime=GuidanceRuntime(verbose=True),
    )
    pd.testing.assert_frame_equal(InternalDataFrame.from_records([
        {
            'text': "Apple's latest product, the iPhone 15, was released in September 2023.",
            'skill_person': "",  # No person mentioned
            'skill_product': 'iPhone 15',
            'skill_date': 'September 2023',
            'skill_location': ""  # No location mentioned
        },
        {
            'text': 'Barack Obama was the 44th president of the United States.',
            'skill_person': 'Barack Obama',
            'skill_product': "",  # No product mentioned
            'skill_date': "",  # No date mentioned
            'skill_location': 'United States'
        }
    ]), predictions)
