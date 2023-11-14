import pandas as pd

from utils import patching, PatchedCalls


@patching(
    target_function=PatchedCalls.OPENAI_MODEL_LIST.value,
    data=[{'input': {}, 'output': {'data': [{'id': 'gpt-3.5-turbo-instruct'}]}}],
)
@patching(
    target_function=PatchedCalls.GUIDANCE.value,
    data=[{
        # apply "Extract named entities" -> Produce "skill_0" column
        'input': {"text_": "Barack Obama was the 44th president of the United States."},
        'output': {"skill_0": "\n- Barack Obama (person)\n- 44th (ordinal number)\n- president (title)\n- United States (location)"}
    }, {
        'input': {"text_": "Apple's latest product, the iPhone 15, was released in September 2023."},
        'output': {"skill_0": '\n- Apple (company)\n- iPhone 15 (product)\n- September 2023 (date)'}
    }, {
        # apply "Translate to French" -> Produce "skill_1" column
        'input': {"text_": "Barack Obama was the 44th president of the United States.", "skill_0": "\n- Barack Obama (person)\n- 44th (ordinal number)\n- president (title)\n- United States (location)"},
        'output': {"skill_1": '\n- Barack Obama (personne)\n- 44e (numéro ordinal)\n- président (titre)\n- États-Unis (emplacement)'}
    }, {
        'input': {"text_": "Apple's latest product, the iPhone 15, was released in September 2023.", "skill_0": '\n- Apple (company)\n- iPhone 15 (product)\n- September 2023 (date)'},
        'output': {"skill_1": '\n- Apple (entreprise)\n- iPhone 15 (produit)\n- Septembre 2023 (date)'}
    }, {
        # apply "Create a structured output in JSON format" -> Produce "skill_2" column
        'input': {"text_": "Barack Obama was the 44th president of the United States.", "skill_0": "\n- Barack Obama (person)\n- 44th (ordinal number)\n- president (title)\n- United States (location)", "skill_1": '\n- Barack Obama (personne)\n- 44e (numéro ordinal)\n- président (titre)\n- États-Unis (emplacement)'},
        'output': {'skill_2': '\n{\n    "personne": "Barack Obama",\n    "numéro ordinal": "44e",\n    "titre": "président",\n    "emplacement": "États-Unis"\n}'}
    }, {
        'input': {"text_": "Apple's latest product, the iPhone 15, was released in September 2023.", "skill_0": '\n- Apple (company)\n- iPhone 15 (product)\n- September 2023 (date)', 'skill_1': '\n- Apple (entreprise)\n- iPhone 15 (produit)\n- Septembre 2023 (date)'},
        'output': {'skill_2': '\n{\n  "entreprise": "Apple",\n  "produit": "iPhone 15",\n  "date": "Septembre 2023"\n}'}
    },],
    strict=False
)
def test_llm_linear_skillset():
    from adala.skills import LinearSkillSet, TransformSkill
    from adala.runtimes import GuidanceRuntime

    skillset = LinearSkillSet(
        skills=[
            TransformSkill(
                name="skill_0",
                instructions="Extract named entities",
                input_template="Input: {text}",
                output_template="Output: {skill_0}"
            ),
            TransformSkill(
                name="skill_1",
                instructions="Translate to French",
                input_template="Input: {skill_0}",
                output_template="Output: {skill_1}"
            ),
            TransformSkill(
                name="skill_2",
                instructions="Create a structured output in JSON format",
                input_template="Input: {skill_1}",
                output_template="Output: {skill_2}"
            ),
        ]
    )
    df = InternalDataFrame([
        "Barack Obama was the 44th president of the United States.",
        "Apple's latest product, the iPhone 15, was released in September 2023.",
        # "The Louvre Museum in Paris houses the Mona Lisa."
    ], columns=["text"])
    predictions = skillset.apply(
        input=df,
        runtime=GuidanceRuntime(),
    )

    pd.testing.assert_frame_equal(InternalDataFrame.from_records([
        # FIRST ROW
        {'text': 'Barack Obama was the 44th president of the United States.',
         'skill_0': '\n- Barack Obama (person)\n- 44th (ordinal number)\n- president (title)\n- United States (location)',
         'skill_1': '\n- Barack Obama (personne)\n- 44e (numéro ordinal)\n- président (titre)\n- États-Unis (emplacement)',
         'skill_2': '\n{\n    "personne": "Barack Obama",\n    "numéro ordinal": "44e",\n    "titre": "président",\n    "emplacement": "États-Unis"\n}'},
        # SECOND ROW
        {'text': "Apple's latest product, the iPhone 15, was released in September 2023.",
         'skill_0': '\n- Apple (company)\n- iPhone 15 (product)\n- September 2023 (date)',
         'skill_1': '\n- Apple (entreprise)\n- iPhone 15 (produit)\n- Septembre 2023 (date)',
         'skill_2': '\n{\n  "entreprise": "Apple",\n  "produit": "iPhone 15",\n  "date": "Septembre 2023"\n}'},
        # THIRD ROW
        # {'text': 'The Louvre Museum in Paris houses the Mona Lisa.',
        #  'skill_0': '\n- The Louvre Museum (Organization)\n- Paris (Location)\n- Mona Lisa (Artwork)',
        #  'skill_1': "\n- Le Musée du Louvre (Organisation)\n- Paris (Lieu)\n- La Joconde (Œuvre d'art)",
        #  'skill_2': '\n{\n    "Organisation": "Le Musée du Louvre",\n    "Lieu": "Paris",\n    "Œuvre d\'art": "La Joconde"\n}'}
    ]), predictions)
