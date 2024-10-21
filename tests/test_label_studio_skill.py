import pytest
import os
import pandas as pd
from adala.agents import Agent



@pytest.mark.asyncio
@pytest.mark.vcr
async def test_label_studio_skill():

    df = pd.DataFrame([
        {"title": "I can't login", "description": "I can't login to the platform"},
        {"title": "Support new file types", "description": "It would be great if we could upload files of type .docx"},
    ])

    agent_payload = {
        "runtimes": {
            "default": {
                "type": "AsyncLiteLLMChatRuntime",
                "model": "gpt-4o-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "max_tokens": 200,
                "temperature": 0,
                "batch_size": 100,
                "timeout": 10,
                "verbose": False,
            }
        },
        "skills": [
            {
                "type": "LabelStudioSkill",
                "name": "AnnotationResult",
                "input_template": '''
                    Given the github issue title:\n{title}\n and the description:\n{description}\n, 
                    classify the issue. Provide a rationale for your classification. 
                    Evaluate the final classification on a Likert scale from 1 to 5, 
                    where 1 is "Completely irrelevant" and 5 is "Completely relevant".''',
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
                """
            }
        ],
    }

    agent = Agent(**agent_payload)
    predictions = await agent.arun(df)

    assert predictions.classification.tolist() == ["Bug report", "Feature request"]
    assert predictions.evaluation.tolist() == [5, 5]
    assert predictions.rationale.tolist() == [
        "The issue clearly indicates a problem with the login functionality of the platform, which is a critical feature. Users are unable to access their accounts, suggesting a potential bug that needs to be addressed.",
        "The issue is requesting the addition of support for a new file type (.docx), which indicates a desire for new functionality in the system. This aligns with the definition of a feature request, as it seeks to enhance the capabilities of the application."
    ]
