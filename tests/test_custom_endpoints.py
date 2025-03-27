import asyncio
import pytest
import os
import pandas as pd
from adala.agents import Agent
from label_studio_sdk.label_interface import LabelInterface
from label_studio_sdk.label_interface.objects import PredictionValue
from adala.runtimes.base import CostEstimate


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_custom_endpoint_simple():

    df = pd.DataFrame(
        [
            {"title": "I can't login", "description": "I can't login to the platform"},
            {
                "title": "Support new file types",
                "description": "It would be great if we could upload files of type .docx",
            },
        ]
    )

    agent_payload = {
        "runtimes": {
            "default": {
                "type": "AsyncLiteLLMChatRuntime",
                "model": "deepseek-ai/DeepSeek-V3-0324-fast",
                "provider": "Custom",
                "base_url": "https://router.huggingface.co/nebius/v1",
                "api_key": os.environ.get("HF_TOKEN"),
            }
        },
        "skills": [
            {
                "type": "LabelStudioSkill",
                "name": "AnnotationResult",
                "input_template": """
                    Given the github issue title:\n{title}\n and the description:\n{description}\n, 
                    classify the issue. Provide a rationale for your classification. 
                    Evaluate the final classification on a Likert scale from 1 to 5, 
                    where 1 is "Completely irrelevant" and 5 is "Completely relevant".""",
                "label_config": """
                <View>
                    <Header value="GitHub Issue Classification"/>
                    <View style="border: 1px solid #ccc; padding: 10px; margin: 10px 0;">
                        <Text name="title" toName="title"/>
                        <TextArea name="description" toName="description"/>
                    </View>
                    <Choices name="classification" toName="title" required="true">
                        <Choice value="Bug report"/>
                        <Choice value="Feature request"/>
                        <Choice value="Question"/>
                        <Choice value="Other"/>
                    </Choices>
                    <TextArea name="rationale" toName="title"/>
                    <Rating name="evaluation" toName="title" maxRating="5" required="true"/>
                </View>
                """,
            }
        ],
    }

    agent = Agent(**agent_payload)
    predictions = await agent.arun(df)

    # Check that we receive valid classifications
    assert all(
        pred in ["Bug report", "Feature request", "Question", "Other"]
        for pred in predictions.classification.tolist()
    )

    # Check that we have rationales for each item
    assert all(len(rationale) > 0 for rationale in predictions.rationale.tolist())

    # Check that we have evaluations and they're all within the valid range
    assert all(1 <= eval_score <= 5 for eval_score in predictions.evaluation.tolist())

    # Assert specific values in the predictions
    expected_classifications = ["Bug report", "Feature request"]
    assert predictions.classification.tolist() == expected_classifications

    # Assert rationales match expected content
    assert "unable to log in" in predictions.rationale[0]
    assert "new file type (.docx)" in predictions.rationale[1]

    # Assert all evaluations are 5
    assert all(score == 5 for score in predictions.evaluation.tolist())

    # Assert precise token counts
    assert predictions._prompt_tokens.tolist() == [80, 87]
    assert predictions._completion_tokens.tolist() == [76, 78]

    # Assert costs are None
    assert all(cost is None for cost in predictions._prompt_cost_usd.tolist())
    assert all(cost is None for cost in predictions._completion_cost_usd.tolist())
    assert all(cost is None for cost in predictions._total_cost_usd.tolist())

    # Assert title and description are None (as they were input fields)
    assert all(title is None for title in predictions.title.tolist())
    assert all(description is None for description in predictions.description.tolist())
