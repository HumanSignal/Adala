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
async def test_custom_endpoint_with_cost_estimate():

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

    # Test get_cost_estimate function
    runtime = agent.get_runtime()

    # Define test prompt and substitutions for cost estimation
    test_prompt = "Classify the following github issue: Title: {title}, Description: {description}"
    substitutions = [
        {"title": "Login broken", "description": "Can't log in to the application"},
        {"title": "Add PDF support", "description": "Need to support PDF file uploads"},
    ]

    # Get cost estimate with output fields
    output_fields = ["classification", "rationale", "evaluation"]
    cost_estimate = await runtime.get_cost_estimate_async(
        prompt=test_prompt,
        substitutions=substitutions,
        output_fields=output_fields,
        provider="Custom",
    )

    # Verify the cost estimate structure
    assert cost_estimate.is_error
    assert (
        cost_estimate.error_message
        == "Model deepseek-ai/DeepSeek-V3-0324-fast for provider Custom not found."
    )
