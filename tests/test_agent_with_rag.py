import pandas as pd
import os
import pytest


@pytest.mark.vcr
def test_rag_with_openai_chat_completion():
    from adala.agents import Agent  # type: ignore
    from adala.skills import LinearSkillSet, ClassificationSkill, RAGSkill  # type: ignore
    from adala.environments import StaticEnvironment  # type: ignore
    from adala.runtimes import OpenAIChatRuntime  # type: ignore
    from adala.memories import VectorDBMemory  # type: ignore

    memory = VectorDBMemory(
        db_name="emotions", openai_api_key=os.environ.get("OPENAI_API_KEY")
    )

    agent = Agent(
        skills=LinearSkillSet(
            skills=[
                RAGSkill(
                    name="rag",
                    input_template="Text: {input}",
                    output_template="{examples}",
                    rag_input_template="Text: {input}\nEmotions: {emotions}",
                    num_results=2,
                    memory=memory,
                ),
                ClassificationSkill(
                    name="emotions",
                    instructions="Recognize emotions from text.",
                    input_template="Examples:\n\n{examples}\n\nNow recognize:\n\nText: {input}",
                    output_template="Emotions: {prediction}",
                    labels={"prediction": ["happy", "sad", "angry", "neutral"]},
                ),
            ]
        ),
        environment=StaticEnvironment(
            ground_truth_columns={"prediction": "emotions"},
            df=pd.DataFrame(
                {
                    "input": [
                        "I am happy",
                        "I am angry",
                        "I am sad",
                    ],
                    "emotions": [
                        "happy",
                        "angry",
                        "sad",
                    ],
                }
            ),
        ),
        runtimes={"default": OpenAIChatRuntime(model="gpt-3.5-turbo")},
        teacher_runtimes={"openai-teacher": OpenAIChatRuntime(model="gpt-4o")},
        default_teacher_runtime="openai-teacher",
    )
    agent.learn(learning_iterations=2)

    # TODO: @matt-bernstein: Add assertions
