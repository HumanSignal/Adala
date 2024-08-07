import re

import pandas as pd
import pytest
from adala.agents import Agent
from adala.environments import (SimpleCodeValidationEnvironment,
                                StaticEnvironment)
from adala.runtimes import OpenAIChatRuntime
from adala.skills import (AnalysisSkill, ClassificationSkill, LinearSkillSet,
                          OntologyCreator, OntologyMerger, ParallelSkillSet,
                          TransformSkill)
from adala.skills.collection.qa import QuestionAnsweringSkill
from adala.skills.collection.summarization import SummarizationSkill
from adala.skills.collection.translation import TranslationSkill
from datasets import load_dataset


@pytest.mark.vcr
def test_classification_skill():
    df = pd.DataFrame(
        [
            {"text": "Apple product with a sleek design.", "category": "Electronics"},
            {
                "text": "Laptop stand for the kitchen.",
                "category": "Furniture/Home Decor",
            },
            {"text": "Chocolate leather boots.", "category": "Footwear/Clothing"},
            {"text": "Wooden cream for surfaces.", "category": "Furniture/Home Decor"},
            {
                "text": "Natural finish for your lips.",
                "category": "Beauty/Personal Care",
            },
        ]
    )

    agent = Agent(
        skills=ClassificationSkill(
            name="product_category_classification",
            input_template="Text: {text}",
            output_template="Category: {predicted_category}",
            labels={
                "predicted_category": [
                    "Footwear/Clothing",
                    "Electronics",
                    "Food/Beverages",
                    "Furniture/Home Decor",
                    "Beauty/Personal Care",
                ]
            },
        ),
        environment=StaticEnvironment(
            df=df, ground_truth_columns={"predicted_category": "category"}
        ),
        teacher_runtimes={"default": OpenAIChatRuntime(model="gpt-4-turbo")},
    )

    agent.learn()
    assert (
        agent.skills["product_category_classification"].instructions
        == """\
\"Classify the input text into its correct product category based on the description provided. Ensure accuracy by closely examining both the specific items mentioned and the context or intended use described in the text. Prioritize aligning the description with the most relevant category based on its functional use and setting, rather than solely on the type of product mentioned.\""""
    )


@pytest.mark.vcr
def test_parallel_skillset_with_analysis():
    df = pd.DataFrame(
        [
            {
                "payload": '{"outputs": [{"entity_group": "ORG", "score": 0.9994323253631592, "word": "Apple Inc", "start": 0, "end": 9}, {"entity_group": "MISC", "score": 0.997283935546875, "word": "iPhone 14", "start": 24, "end": 33}], "inputs": "Apple Inc. released the iPhone 14 in September 2022, featuring satellite connectivity."}'
            },
            {
                "payload": '{"outputs": [{"entity_group": "MISC", "score": 0.9428057670593262, "word": "Ubuntu", "start": 26, "end": 32}, {"entity_group": "MISC", "score": 0.962793231010437, "word": "Ubuntu", "start": 51, "end": 57}, {"entity_group": "ORG", "score": 0.998673677444458, "word": "Canonical Ltd", "start": 87, "end": 100}], "inputs": "The latest version of the Ubuntu operating system, Ubuntu 22.04, was made available by Canonical Ltd. in April."}'
            },
            {
                "payload": '{"outputs": [{"entity_group": "ORG", "score": 0.979661226272583, "word": "Tesla", "start": 0, "end": 5}, {"entity_group": "ORG", "score": 0.8453200459480286, "word": "Cybertru", "start": 12, "end": 20}, {"entity_group": "MISC", "score": 0.7452507019042969, "word": "##ck", "start": 20, "end": 22}, {"entity_group": "PER", "score": 0.9728273153305054, "word": "El", "start": 78, "end": 80}, {"entity_group": "PER", "score": 0.9739447236061096, "word": "##on Musk", "start": 80, "end": 87}], "inputs": "Tesla\'s new Cybertruck is set to hit the roads in late 2023, according to CEO Elon Musk."}'
            },
            {
                "payload": '{"outputs": [{"entity_group": "ORG", "score": 0.9987253546714783, "word": "Google", "start": 0, "end": 6}, {"entity_group": "ORG", "score": 0.9994670748710632, "word": "Alphabet Inc", "start": 25, "end": 37}, {"entity_group": "MISC", "score": 0.9959796667098999, "word": "Pixel 6", "start": 91, "end": 98}], "inputs": "Google\'s parent company, Alphabet Inc., saw a rise in stock prices after the launch of the Pixel 6."}'
            },
            {
                "payload": '{"outputs": [{"entity_group": "ORG", "score": 0.999211311340332, "word": "Samsung Electronics", "start": 0, "end": 19}, {"entity_group": "ORG", "score": 0.9967896342277527, "word": "LG Display", "start": 38, "end": 48}, {"entity_group": "MISC", "score": 0.47527530789375305, "word": "O", "start": 56, "end": 57}, {"entity_group": "MISC", "score": 0.5774009227752686, "word": "##D", "start": 59, "end": 60}], "inputs": "Samsung Electronics is competing with LG Display in the OLED market."}'
            },
        ]
    )

    skillset = ParallelSkillSet(
        skills=[
            AnalysisSkill(
                name="code_generation",
                input_template="Input JSON: {payload}",
                output_template="Code: {code}",
                instructions="""
    Format description: 
    id - Identifier for the labeling task from the dataset.
    data - Data dict copied from the input data task format.
    project - Identifier for a specific project in Label Studio.
    predictions - Array containing the labeling results for the task.
    predictions.id - Identifier for the completed task.
    predictions.lead_time - Time in seconds to label the task.
    predictions.result - Array containing the results of the labeling or annotation task.
    result.id - Identifier for the specific annotation result for this task.
    result.from_name - Name of the tag used to label the region. See control tags.
    result.to_name	- Name of the object tag that provided the region to be labeled. See object tags.
    result.type	- Type of tag used to annotate the task.
    result.value - Tag-specific value that includes details of the result of labeling the task. The value - structure depends on the tag for the label. For more information, see Explore each tag.
    predictions.score - The overall score of the result, based on the probabilistic output, confidence level, or other.

    Following the target JSON format provided, write a minimal python code that transform input json to this format. \
    Assume the input data will be read from the standard input (stdin), and the output generated will be directed to the standard output (stdout).""",
            )
        ]
    )

    env = SimpleCodeValidationEnvironment(df=df, code_fields={"code": "payload"})

    agent = Agent(
        skills=skillset,
        environment=env,
        runtimes={"default": OpenAIChatRuntime(model="gpt-4o")},
        teacher_runtimes={"default": OpenAIChatRuntime(model="gpt-4o")},
    )
    # AnalysisSkill.improve not implemented
    # agent.learn(learning_iterations=1, num_feedbacks=1, batch_size=3)
    predictions = agent.run()
    expected_code = """\
import sys
import json

# Read input from stdin
input_data = sys.stdin.read()

# Parse the input JSON
input_json = json.loads(input_data)

# Initialize the output structure
output_json = {
    "id": None,
    "data": input_json,
    "project": None,
    "predictions": [
        {
            "id": None,
            "lead_time": None,
            "result": [],
            "score": None
        }
    ]
}

# Populate the result field
for entity in input_json.get("outputs", []):
    result_entry = {
        "id": None,
        "from_name": "label",
        "to_name": "text",
        "type": "labels",
        "value": {
            "start": entity["start"],
            "end": entity["end"],
            "text": entity["word"],
            "labels": [entity["entity_group"]]
        }
    }
    output_json["predictions"][0]["result"].append(result_entry)

# Output the transformed JSON
print(json.dumps(output_json, indent=4))"""
    # temp hack to compare strings properly
    expected_code = expected_code.replace("'\n'", "'\\n'")
    assert predictions.code[0] == expected_code


@pytest.mark.vcr
def test_summarization_skill():
    df = pd.DataFrame(
        [
            "Caffeine, found in coffee beans and synthesized in labs, has the same structure in various forms. It is a stimulant that enhances physical strength and mental stimulation. Habitual use is linked to reduced risks of diseases. Caffeine works by antagonizing adenosine receptors, promoting alertness by inhibiting sedation. It affects neurotransmitter systems like dopamine and serotonin.",
            "Vitamin C is a water-soluble essential vitamin found in fruits and vegetables, crucial for immune system function and collagen synthesis. It is commonly supplemented during colds to reduce duration. Studies suggest doses of 200-2000 mg are beneficial. Vitamin C is studied for its antioxidant effects in Alzheimer's disease and cancer, with potential benefits in reducing oxidative stress and promoting cytotoxic effects in cancer cells. Further research and human studies are needed to determine efficacy.",
            "Vitamin D is a fat-soluble nutrient essential for human survival. It is primarily obtained from the sun, oily fish, eggs, and fortified foods like milk. Supplementing with vitamin D can improve immune health, bone health, and overall well-being. It may also reduce the risk of cancer, diabetes, and multiple sclerosis. The effects of vitamin D depend on the levels of 25-hydroxyvitamin D in the blood, and benefits are most noticeable when reversing a deficiency.",
        ],
        columns=["text"],
    )

    agent = Agent(
        skills=SummarizationSkill(name="summarization", input_data_field="text")
    )

    predictions = agent.run(df)
    assert (
        predictions.summary[0]
        == """\
Caffeine, found in coffee beans and synthesized in labs, has the same structure in various forms. It is a stimulant that improves physical strength and mental stimulation, classified as a nootropic. Habitual use is linked to reduced risks of Alzheimer's, cirrhosis, and liver cancer. Caffeine antagonizes adenosine receptors, promoting alertness by inhibiting sedation. It affects dopamine, serotonin, acetylcholine, and adrenaline systems."""
    )
    assert (
        predictions.summary[2]
        == """\
Vitamin D is a fat-soluble nutrient essential for human survival. It is primarily obtained from the sun, oily fish, eggs, and fortified foods like milk. Supplementing with vitamin D can improve immune health, bone health, and overall well-being. It may also reduce the risk of cancer, diabetes, and multiple sclerosis. The benefits of vitamin D depend on the individual's levels of 25-hydroxyvitamin D, and improvements are typically observed when reversing a deficiency."""
    )

    pd.testing.assert_series_equal(predictions.text, df.text)


@pytest.mark.vcr
@pytest.mark.skip(reason="flakes in CI")
def test_transform_skill():
    def extract_and_convert_numbers(text):
        pattern = "\d{1,3}(?:,\d{3})*\.?\d*"
        numbers = re.findall(pattern, text)
        return [
            int(num.replace(",", "").split(".")[0])
            for num in numbers
            if num.replace(",", "").split(".")[0]
        ]

    def evaluate_answers(ground_truth, prediction):
        pred = extract_and_convert_numbers(prediction)
        gt = extract_and_convert_numbers(ground_truth)
        return any(p == gt[-1] for p in pred)

    gsm8k = load_dataset("gsm8k", "main")

    df_train = pd.DataFrame(
        {"question": gsm8k["train"]["question"], "answer": gsm8k["train"]["answer"]}
    )
    df_test = pd.DataFrame(
        {"question": gsm8k["test"]["question"], "answer": gsm8k["test"]["answer"]}
    )
    # take a subset for faster testing
    df_test = df_test.iloc[:10]

    skills = LinearSkillSet(
        skills=[
            TransformSkill(
                name="math_solver",
                # we start with no instructions then explain how agent can learn more details
                instructions="",
                # instructions=prompt,
                input_template="Q: {question}",
                # here is the baseline established in Kojima et al., 2022 paper
                # output_template='A: Let’s think step by step. {rationale}\nFinal numerical answer:{answer}',
                output_template="A: {answer}",
                instructions_first=False,
            )
        ]
    )

    agent = Agent(
        skills=skills,
        # this is where agent receives the ground truth signal
        environment=StaticEnvironment(df=df_train, matching_function=evaluate_answers),
        teacher_runtimes={"gpt4": OpenAIChatRuntime(model="gpt-4-1106-preview")},
        default_teacher_runtime="gpt4",
    )

    result_baseline = agent.run(df_test.drop(columns="answer"))


@pytest.mark.vcr
@pytest.mark.skip(reason="flakes in CI")
def test_ontology_creator_merger_skill():
    ds = load_dataset("ag_news", "default")
    df = pd.DataFrame(data=ds["train"][:10]["text"], columns=["text"])

    target = (
        "help analysts identify the primary reason for changes in the stock market."
    )

    a = Agent(
        skills=LinearSkillSet(
            skills=[OntologyCreator(target=target), OntologyMerger(target=target)]
        ),
        runtimes={"gpt4": OpenAIChatRuntime(model="gpt-4-1106-preview", verbose=False)},
        default_runtime="gpt4",
    )

    pred = a.run(df)


@pytest.mark.vcr
def test_question_answering_skill():
    df = pd.DataFrame(
        [
            {
                "question": "In quantum mechanics, what principle asserts that it's impossible to simultaneously know the exact position and momentum of a particle?",
                "expected_answer": "Heisenberg Uncertainty Principle",
            },
            {
                "question": "Which famous poet wrote 'The Love Song of J. Alfred Prufrock'?",
                "expected_answer": "T.S. Eliot",
            },
            {
                "question": "What mathematical theorem states that in any right-angled triangle, the area of the square whose side is the hypotenuse is equal to the sum of the areas of the squares whose sides are the two legs?",
                "expected_answer": "Pythagorean Theorem",
            },
            {
                "question": "Which philosophical paradox involves a ship where all of its wooden parts are replaced with metal parts?",
                "expected_answer": "Ship of Theseus",
            },
            {
                "question": "In the world of programming, what is the design principle that suggests a system should be open for extension but closed for modification?",
                "expected_answer": "Open/Closed Principle",
            },
        ]
    )

    agent = Agent(skills=QuestionAnsweringSkill())

    predictions = agent.run(df)
    assert (predictions.answer == predictions.expected_answer).mean() == 3 / 5


@pytest.mark.vcr
def test_linear_skillset():
    df = pd.DataFrame(
        [
            {
                "category": "Macronutrients",
                "entities": "Carbohydrates, Proteins, Fats",
                "text": "Carbohydrates provide quick energy, proteins are essential for muscle repair and growth, and fats are vital for long-term energy storage and cell function.",
            },
            {
                "category": "Vitamins",
                "entities": "Vitamin A, Vitamin C, Vitamin D",
                "text": "Vitamin A is crucial for good vision and a healthy immune system, Vitamin C helps in the repair of tissues and the enzymatic production of certain neurotransmitters, and Vitamin D is essential for strong bones and teeth as it helps the body absorb calcium.",
            },
            {
                "category": "Minerals",
                "entities": "Calcium, Iron, Magnesium",
                "text": "Calcium is necessary for maintaining healthy bones and teeth, Iron is crucial for making red blood cells and transporting oxygen throughout the body, and Magnesium plays a role in over 300 enzyme reactions in the human body, including the metabolism of food, synthesis of fatty acids and proteins, and the transmission of nerve impulses.",
            },
        ]
    )

    agent = Agent(
        # Require agent to learn sequence of two skills
        skills=LinearSkillSet(
            skills=[
                TransformSkill(
                    name="skill_0",
                    instructions="...",
                    input_template="Input: {category}",
                    output_template="Output: {skill_0}",
                ),
                TransformSkill(
                    name="skill_1",
                    instructions="...",
                    input_template="Input: {skill_0}",
                    output_template="Output: {skill_1}",
                ),
            ]
        ),
        # provide ground truth demonstration in environment
        environment=StaticEnvironment(
            df=df,
            ground_truth_columns={"skill_0": "entities", "skill_1": "text"},
            matching_function="fuzzy",
            matching_threshold=0.9,
        ),
        teacher_runtimes={"default": OpenAIChatRuntime(model="gpt-4")},
    )

    agent.learn(learning_iterations=2)
    assert (
        agent.skills["skill_0"].instructions
        == '''\
"Provide specific examples that fall under the given category. If possible, list them in the order of their commonality or importance."'''
    )
    # TODO: not learned with 2 iterations, need to increase learning_iterations
    assert agent.skills["skill_1"].instructions == "..."


@pytest.mark.vcr
def test_translation_skill():
    df = pd.DataFrame(
        [
            {"text": "El sol brilla siempre", "language": "Spanish"},
            {"text": "La vie est belle", "language": "French"},
            {"text": "Der Wald ruft mich", "language": "German"},
            {"text": "Amo la pizza napoletana", "language": "Italian"},
            {"text": "春天的花很美", "language": "Chinese"},
            {"text": "Звезды сверкают ночью", "language": "Russian"},
            {"text": "雨の後の虹", "language": "Japanese"},
            {"text": "커피가 필요해", "language": "Korean"},
            {"text": "A música toca a alma", "language": "Portuguese"},
            {"text": "सपने सच होते हैं", "language": "Hindi"},
        ]
    )

    agent = Agent(skills=TranslationSkill(target_language="Swahili"))

    predictions = agent.run(df)
    assert predictions.translation.tolist() == ["Jua huzidi kung'aa daima", 'Maisha ni mazuri', 'Msitu unaniita', 'Napenda pizza ya Napolitana', 'Maua ya spring ni mazuri sana', "Nyota zinang'aa usiku", 'Upinde wa mvua baada ya mvua', 'Ninahitaji kahawa', 'Muziki huchezesha roho', 'Ndoto zinakuwa kweli']
