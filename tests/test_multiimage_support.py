import pandas as pd
import pytest
from adala.agents import Agent
import asyncio
import os
from unittest.mock import patch, MagicMock, AsyncMock
from pydantic import BaseModel
from typing import Dict, Any, List
import json
from adala.utils.llm_utils import (
    arun_instructor_with_messages,
    arun_instructor_with_payloads,
)
from litellm.types.utils import Usage
from adala.utils.parse import MessageChunkType


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


class SimpleResponse(BaseModel):
    result: str
    output: List[str] = []


@pytest.mark.asyncio
async def test_message_trimming_mocked():
    """Test that messages exceeding context window are properly trimmed and message counts tracked.

    This test uses a real image URL duplicated many times to test the context window trimming.
    Only the create_with_completion method is mocked, all other util functions perform actual computations.
    """
    # Use a real image URL and duplicate it many times (200+) to exceed context window
    real_image_url = "https://htx-pub.s3.amazonaws.com/demo/ocr/pdf/output_0000.png"

    # Create a payload with a large list of images
    image_urls = [real_image_url] * 2000
    payload = {"images": image_urls}

    # Create a user prompt template that references the images
    user_prompt_template = "Analyze these images: {images}. Provide the list of authors and the year of publication."

    # Create mock objects
    mock_client = AsyncMock()

    # Create a realistic completion response similar to the example in test_MIG.yaml
    mock_response = SimpleResponse(
        result="Success",
        output=[
            "Fihn et al., Circulation 2012, 126: e354-471",
            "Amsterdam et al., J Am Coll Cardiol 2014, 64: e139-228",
            "Task Force et al., Eur Heart J 2013, 34: 2949-3003",
        ],
    )

    # Create realistic completion object with usage stats
    mock_completion = MagicMock()
    mock_completion.model = "gpt-4o"
    mock_completion.usage = Usage(
        prompt_tokens=128000,
        completion_tokens=370,
        total_tokens=128370,
    )

    # Mock only the create_with_completion method
    mock_client.chat.completions.create_with_completion = AsyncMock(
        return_value=(mock_response, mock_completion)
    )

    # Run the function with our minimally mocked dependencies
    # Using the higher-level function that handles payloads
    results = await arun_instructor_with_payloads(
        client=mock_client,
        payloads=[payload],
        user_prompt_template=user_prompt_template,
        response_model=SimpleResponse,
        model="gpt-4o",
        input_field_types={"images": MessageChunkType.IMAGE_URLS},
        extra_fields={},
        ensure_messages_fit_in_context_window=True,
        split_into_chunks=True,
    )

    results = results[0]
    # Log the result message counts for debugging
    print(f"Original message count: {len(image_urls)}")
    print(f"Message counts in result: {results['_message_counts']}")

    # Verify the message_counts in the result
    assert "_message_counts" in results

    # The message counts should include image_url and text
    assert "image_url" in results["_message_counts"]
    assert "text" in results["_message_counts"]

    # Verify that the number of images was trimmed from 2000 to 1391 (each image is around 100 tokens so this is the max that fits in the context window of 128000 tokens)
    assert results["_message_counts"]["image_url"] == 1391

    # Verify that the number of text messages was also trimmed
    assert results["_message_counts"]["text"] == 1  # last message is also trimmed

    # Verify that the metrics are included
    assert results["_prompt_tokens"] == 128000
    assert results["_completion_tokens"] == 370
    assert results["_prompt_cost_usd"] == 0.32
    assert results["_completion_cost_usd"] == 0.0037
    assert results["_total_cost_usd"] == 0.3237
