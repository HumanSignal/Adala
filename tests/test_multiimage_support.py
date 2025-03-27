import pandas as pd
import pytest
from adala.agents import Agent
import asyncio
import os


@pytest.mark.vcr
def test_MIG():
    df = pd.DataFrame(
        [
            {
                "pdf": [
                    "https://htx-pub.s3.amazonaws.com/demo/ocr/pdf/output_0000.png",
                    "https://htx-pub.s3.amazonaws.com/demo/ocr/pdf/output_0001.png",
                    "https://htx-pub.s3.amazonaws.com/demo/ocr/pdf/output_0002.png",
                    "https://htx-pub.s3.amazonaws.com/demo/ocr/pdf/output_0003.png",
                    "https://htx-pub.s3.amazonaws.com/demo/ocr/pdf/output_0004.png",
                    "https://htx-pub.s3.amazonaws.com/demo/ocr/pdf/output_0005.png",
                    "https://htx-pub.s3.amazonaws.com/demo/ocr/pdf/output_0006.png",
                    "https://htx-pub.s3.amazonaws.com/demo/ocr/pdf/output_0007.png",
                    "https://htx-pub.s3.amazonaws.com/demo/ocr/pdf/output_0008.png",
                ]
            }
        ]
    )

    agent_payload = {
        "runtimes": {
            "default": {
                "type": "AsyncLiteLLMVisionRuntime",
                "provider": "azure",
                "model": "azure/gpt4o",
                "base_url": "https://humansignal-openai-test.openai.azure.com/",
                "api_key": os.getenv("AZURE_OPENAI_API_KEY", "123"),
                "max_tokens": 4096,
                # TODO: All other providers are working only with `pytest --record-mode=overwrite -k test_MIG` (live calls)
                # as their internal calls contain syc https requests that are not recorded by vcrpy
                # "provider": "vertex_ai",
                # "model": "vertex_ai/gemini-2.0-flash-exp",
                # 'vertex_credentials': os.getenv("VERTEXAI_CREDENTIALS"),
                # "provider": "gemini",
                # "model": "gemini/gemini-2.0-flash-exp",
                # "api_key": os.getenv("GEMINI_API_KEY"),
                "instructor_mode": "json_mode",
            }
        },
        "skills": [
            {
                "type": "LabelStudioSkill",
                "name": "MIG",
                "input_template": """
                    Extract the publication reference with author name(s) and year from the last page 9: {pdf}. DON'T ANALYZE ANY OTHER PAGES.
                """,
                "label_config": """
                <View>
                    <Style>
                    .container {
                    display: flex;
                    align-items: flex-start;
                    height: 100vh;
                    overflow: hidden;
                    }

                    .left {
                    width: 50%;
                    padding: 1rem;
                    border-right: 1px solid #ccc;
                    }

                    .right {
                    width: 50%;
                    padding: 1rem;
                    height: 100%;
                    overflow-y: auto;
                    }
                </Style>
                <View className="container">
                    <View className="left">
                    <Image valueList="$pdf" name="pdf"/>
                    </View>
                <View className="right">
                <TextArea name="output" toName="pdf" rows="5" maxSubmissions="1" showSubmitButton="false" editable="true"/>
                    </View>
                </View>
                </View>
                """,
            }
        ],
    }

    agent = Agent(**agent_payload)
    predictions = asyncio.run(agent.arun(df))

    outputs = predictions.output.tolist()
    assert isinstance(outputs[0], list)
    assert any("Fihn et al., Circulation 2012" in item for item in outputs[0])
    
    assert predictions._prompt_tokens[0] == 7087
    assert predictions._completion_tokens[0] == 370
    assert predictions._prompt_cost_usd[0] == 0.035435
    assert predictions._completion_cost_usd[0] == 0.00555
    assert predictions._total_cost_usd[0] == 0.040985
