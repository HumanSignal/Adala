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
        == 'Classify the input text into the correct product category by emphasizing both the primary function of the item and its intended context of use. Even if certain keywords might suggest a typical category, analyze how the context or specific usage mentioned might indicate a different category. Analyze the entire text holistically to understand its full context and primary purpose before deciding on the category.'
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
    "id": "",
    "data": {
        "text": input_json["inputs"]
    },
    "project": "",
    "predictions": [
        {
            "id": "",
            "lead_time": 0,
            "result": [
                {
                    "id": str(i),
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels",
                    "value": {
                        "start": entity["start"],
                        "end": entity["end"],
                        "score": entity["score"],
                        "text": entity["word"],
                        "labels": [entity["entity_group"]]
                    }
                } for i, entity in enumerate(input_json["outputs"])
            ],
            "score": 0
        }
    ]
}

# Output the transformed JSON
print(json.dumps(output_json, indent=4))"""
    assert predictions.code[0] == expected_code


@pytest.mark.vcr
def test_summarization_skill():
    df = pd.DataFrame(
        [
            "Caffeine comes from coffee beans, but it can also be synthesized in a laboratory. It has the same structure whether it’s in coffee, energy drinks, tea, or pills. Caffeine is a powerful stimulant, and it can be used to improve physical strength and endurance. It is classified as a nootropic because it sensitizes neurons and provides mental stimulation. Habitual caffeine use is also associated with a reduced risk of Alzheimer's disease, cirrhosis, and liver cancer. Caffeine’s main mechanism concerns antagonizing adenosine receptors. Adenosine causes sedation and relaxation when it acts upon its receptors, located in the brain. Caffeine prevents this action and causes alertness and wakefulness. This inhibition of adenosine can influence the dopamine, serotonin, acetylcholine, and adrenaline systems. For practical tips on the optimal use of caffeine, check out our Supplement Guides.",
            "Vitamin C is a water-soluble essential vitamin that can be found in fruits and vegetables, especially citrus. Humans are unable to synthesize vitamin C from their bodies, so it must be acquired through dietary intake. Vitamin C is important for immune system function and is a powerful antioxidant. It also acts as a cofactor for collagen synthesis.[2]. People often supplement with vitamin C when they have a cold. According to various studies, vitamin C may be effective in reducing the duration of a cold, but does not seem to reduce the frequency of colds in a population.[3][4] The available literature suggests that a dose ranging from 200 mg to 2,000 mg could be beneficial for reducing cold duration.Often utilized for its antioxidant effects, vitamin C has been studied for its potential role in Alzheimer’s disease and cancer. Lower vitamin C levels are present in people with Alzheimer’s, even with adequate dietary intake.[5] It is thought that oxidative stress plays a major role in the pathogenesis of the disease, so vitamin C’s antioxidative effects could be beneficial.[6][7] In rodent studies, oral vitamin C was able to reduce oxidative and inflammatory biomarkers.[8] In recent cancer research, vitamin C was found to promote oxidative stress in cancer cells, leading to cytotoxic effects at high doses in mice.[9] While promising, further research and human studies are required to determine efficacy.",
            "Vitamin D is a fat-soluble nutrient. It is one of the 24 micronutrients critical for human survival. The sun is the major natural source through eliciting vitamin D production in the skin, but vitamin D is also found naturally in oily fish and eggs and is added to milk and milk alternatives. Supplemental vitamin D is associated with a range of benefits, including improved immune health, bone health, and well-being. Supplementation may also reduce the risk of cancer mortality, diabetes, and multiple sclerosis.The effects of vitamin D likely depend on a person’s circulating levels of 25-hydroxyvitamin D (25(OH)D; a form of vitamin D that is measured in blood samples to determine vitamin D status), and many of its benefits will only be seen when a deficiency is reversed.",
        ],
        columns=["text"],
    )

    agent = Agent(
        skills=SummarizationSkill(name="summarization", input_data_field="text")
    )

    predictions = agent.run(df)
    # breakpoint()
    assert (
        predictions.text[0]
        == """\
Caffeine comes from coffee beans, but it can also be synthesized in a laboratory. It has the same structure whether it’s in coffee, energy drinks, tea, or pills. Caffeine is a powerful stimulant, and it can be used to improve physical strength and endurance. It is classified as a nootropic because it sensitizes neurons and provides mental stimulation. Habitual caffeine use is also associated with a reduced risk of Alzheimer's disease, cirrhosis, and liver cancer. Caffeine’s main mechanism concerns antagonizing adenosine receptors. Adenosine causes sedation and relaxation when it acts upon its receptors, located in the brain. Caffeine prevents this action and causes alertness and wakefulness. This inhibition of adenosine can influence the dopamine, serotonin, acetylcholine, and adrenaline systems. For practical tips on the optimal use of caffeine, check out our Supplement Guides."""
    )
    assert (
        predictions.text[2]
        == """\
Vitamin D is a fat-soluble nutrient. It is one of the 24 micronutrients critical for human survival. The sun is the major natural source through eliciting vitamin D production in the skin, but vitamin D is also found naturally in oily fish and eggs and is added to milk and milk alternatives. Supplemental vitamin D is associated with a range of benefits, including improved immune health, bone health, and well-being. Supplementation may also reduce the risk of cancer mortality, diabetes, and multiple sclerosis.The effects of vitamin D likely depend on a person’s circulating levels of 25-hydroxyvitamin D (25(OH)D; a form of vitamin D that is measured in blood samples to determine vitamin D status), and many of its benefits will only be seen when a deficiency is reversed."""
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
        == '"Given a category, directly provide a list of the most common or well-known items that belong to that category. Do not provide a definition or repeat the category."'
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

    assert predictions.translation.tolist() == ["Jua huzidi kung'aa daima",
                                                'Maisha ni mazuri',
                                                'Msitu unaniita',
                                                'Napenda pizza ya Napolitana',
                                                'Maua ya spring ni mazuri',
                                                "Nyota zinang'aa usiku",
                                                'Upinde wa mvua baada ya mvua',
                                                'Nahitaji kahawa',
                                                'Muziki huchezesha roho',
                                                'Ndoto zinakuwa kweli']


@pytest.mark.vcr
def test_entity_extraction():
    from adala.skills.collection.entity_extraction import EntityExtraction
    # documents that contain entities
    df = pd.DataFrame([
        {
            "text": "Apple Inc. is an American multinational technology company that specializes in consumer electronics, computer software, and online services."},
        {"text": "The iPhone 14 is the latest smartphone from Apple Inc."},
        {"text": "The MacBook Pro is a line of Macintosh portable computers introduced in January 2006 by Apple Inc."},
        {"text": "The Apple Watch is a line of smartwatches produced by Apple Inc."},
        {"text": "The iPad is a line of tablet computers designed, developed, and marketed by Apple Inc."},
    ])

    agent = Agent(
        skills=EntityExtraction(labels=["Organization", "Person", "Product", "Version"]),
        runtimes={'default': OpenAIChatRuntime(model="gpt-4o-mini")}
    )
    predictions = agent.run(df)
    assert predictions.entities.tolist() == [[{'quote_string': 'Apple Inc.',
                                               'label': 'Organization',
                                               'start': 0,
                                               'end': 10}],
                                             [{'quote_string': 'iPhone 14', 'label': 'Product', 'start': 4, 'end': 13},
                                              {'quote_string': 'Apple Inc.',
                                               'label': 'Organization',
                                               'start': 44,
                                               'end': 54}],
                                             [{'quote_string': 'MacBook Pro', 'label': 'Product', 'start': 4,
                                               'end': 15},
                                              {'quote_string': 'Macintosh', 'label': 'Product', 'start': 29, 'end': 38},
                                              {'quote_string': 'January 2006', 'label': 'Version', 'start': 72,
                                               'end': 84},
                                              {'quote_string': 'Apple Inc.',
                                               'label': 'Organization',
                                               'start': 88,
                                               'end': 98}],
                                             [{'quote_string': 'Apple Watch', 'label': 'Product', 'start': 4,
                                               'end': 15},
                                              {'quote_string': 'Apple Inc.',
                                               'label': 'Organization',
                                               'start': 54,
                                               'end': 64}],
                                             [{'quote_string': 'iPad', 'label': 'Product', 'start': 4, 'end': 8},
                                              {'quote_string': 'Apple Inc.',
                                               'label': 'Organization',
                                               'start': 76,
                                               'end': 86}]]


@pytest.mark.vcr
def test_entity_extraction_no_labels():
    from adala.skills.collection.entity_extraction import EntityExtraction
    # documents that contain entities
    df = pd.DataFrame([
        {
            "text": "Apple Inc. is an American multinational technology company that specializes in consumer electronics, computer software, and online services."},
        {"text": "The iPhone 14 is the latest smartphone from Apple Inc."},
        {"text": "The MacBook Pro is a line of Macintosh portable computers introduced in January 2006 by Apple Inc."},
        {"text": "The Apple Watch is a line of smartwatches produced by Apple Inc."},
        {"text": "The iPad is a line of tablet computers designed, developed, and marketed by Apple Inc."},
    ])

    agent = Agent(
        skills=EntityExtraction(
            input_template='Extract entities from the input text that represents the main points of discussion.\n\nInput:\n"""\n{text}\n"""',
        ),
        runtimes={'default': OpenAIChatRuntime(model="gpt-4o")}
    )
    predictions = agent.run(df)
    assert predictions.entities.tolist() == [[{'quote_string': 'Apple Inc.', 'start': 0, 'end': 10},
                                              {'quote_string': 'American multinational technology company',
                                               'start': 17,
                                               'end': 58},
                                              {'quote_string': 'consumer electronics', 'start': 79, 'end': 99},
                                              {'quote_string': 'computer software', 'start': 101, 'end': 118},
                                              {'quote_string': 'online services', 'start': 124, 'end': 139}],
                                             [{'quote_string': 'iPhone 14', 'start': 4, 'end': 13},
                                              {'quote_string': 'Apple Inc.', 'start': 44, 'end': 54}],
                                             [{'quote_string': 'The MacBook Pro', 'start': 0, 'end': 15},
                                              {'quote_string': 'Macintosh portable computers', 'start': 29, 'end': 57},
                                              {'quote_string': 'January 2006', 'start': 72, 'end': 84},
                                              {'quote_string': 'Apple Inc.', 'start': 88, 'end': 98}],
                                             [{'quote_string': 'The Apple Watch', 'start': 0, 'end': 15},
                                              {'quote_string': 'smartwatches', 'start': 29, 'end': 41},
                                              {'quote_string': 'Apple Inc.', 'start': 54, 'end': 64}],
                                             [{'quote_string': 'iPad', 'start': 4, 'end': 8},
                                              {'quote_string': 'tablet computers', 'start': 22, 'end': 38},
                                              {'quote_string': 'Apple Inc.', 'start': 76, 'end': 86}]]
