import os
import pytest
from pydantic import BaseModel, Field
from instructor import Instructor
from tenacity import Retrying, stop_after_attempt, wait_fixed
from adala.utils.llm_utils import run_instructor_with_messages, count_message_types
from adala.utils.parse import MessageChunkType
from typing import Optional, List


class ImageDescription(BaseModel):
    """Model for image description response."""

    content: str = Field(..., description="The main content of the image")
    objects: List[str] = Field(..., description="List of objects visible in the image")


@pytest.mark.vcr
def test_run_instructor_with_messages_gemini_image():
    """Test run_instructor_with_messages function with Gemini model using text and image messages."""
    # Set up test dependencies
    from adala.runtimes._litellm import InstructorClientMixin

    client = InstructorClientMixin(
        model="gemini/gemini-2.0-flash-exp",
        provider="gemini",
        api_key=os.getenv("GEMINI_API_KEY", "fake_api_key_for_vcr"),
    ).client

    # Configure retry policy (3 attempts with 1 second delay)
    retries = Retrying(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
    )

    # Create sample image URL - this should be a real image URL for actual testing
    # For VCR testing, this is just a placeholder that will be captured in the cassette
    sample_image_url = "https://htx-pub.s3.amazonaws.com/demo/ocr/pdf/output_0000.png"

    # Create messages with combined text and image
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that analyzes images.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please describe this image in detail:"},
                {"type": "image_url", "image_url": {"url": sample_image_url}},
            ],
        },
    ]

    # Call the function being tested
    response = run_instructor_with_messages(
        client=client,
        messages=messages,
        response_model=ImageDescription,
        model="gemini/gemini-2.0-flash-exp",
        canonical_model_provider_string="gemini/gemini-2.0-flash-exp",
        temperature=0.1,
        max_tokens=500,
        retries=retries,
    )

    assert (
        response["content"]
        == "A medical document titled Angina/Chest Pain, containing information about different types of angina and chest pain, including unstable angina, new/worsening angina, stable angina, post-infarction ischemia, and chest pain of unknown etiology. It also includes notes on InterQual criteria, an overview of angina pectoris, acute coronary syndrome, telehealth, and application to specialty referral."
    )
    assert response["objects"] == ["medical document", "angina", "chest pain"]
    assert response["_prompt_tokens"] == 3395
    assert response["_completion_tokens"] == 86
    assert response["_prompt_cost_usd"] == 0
    assert response["_completion_cost_usd"] == 0
    assert response["_total_cost_usd"] == 0
    assert response["_message_counts"] == {"text": 2, "image_url": 1}
    assert response["_inference_time"] > 0


def test_count_message_types():
    """Test the count_message_types class method."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, world!"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this image: "},
                {
                    "type": "image_url",
                    "image_url": {"url": "http://example.com/image.jpg"},
                },
            ],
        },
    ]

    counts = count_message_types(messages)

    assert counts["text"] == 3  # 2 text strings + 1 text chunk
    assert counts["image_url"] == 1


def test_message_type_counting_with_various_inputs():
    """Test count_message_types with various message formats."""
    messages = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "Text message"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Chunked text"},
                {
                    "type": "image_url",
                    "image_url": {"url": "http://example.com/image.jpg"},
                },
            ],
        },
        {"type": "text", "text": "Direct text chunk"},
        {"unknown_format": "This should be counted as text"},
    ]

    counts = count_message_types(messages)

    # 2 text messages + 1 chunked text + 1 direct text chunk + 1 unknown format
    assert counts["text"] == 5
    assert counts["image_url"] == 1
