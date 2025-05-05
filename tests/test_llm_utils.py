import os
import pytest
from pydantic import BaseModel, Field
from instructor import Instructor
from tenacity import Retrying, stop_after_attempt, wait_fixed
from adala.utils.llm_utils import (
    run_instructor_with_messages,
    count_message_types,
    check_model_pdf_support,
)
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
        model="gemini/gemini-2.0-flash",
        canonical_model_provider_string="gemini/gemini-2.0-flash",
        temperature=0.1,
        max_tokens=500,
        retries=retries,
    )

    assert response["content"] == "A medical document about angina and chest pain"
    assert response["objects"] == ["text"]
    assert response["_prompt_tokens"] == 3395
    assert response["_completion_tokens"] == 13
    assert response["_prompt_cost_usd"] == 0.00033949999999999996
    assert response["_completion_cost_usd"] == 5.199999999999999e-06
    assert response["_total_cost_usd"] == 0.0003447
    assert response["_message_counts"] == {"text": 2, "image_url": 1}
    assert response["_inference_time"] > 0


@pytest.mark.vcr
@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["openai/gpt-4o-mini", "anthropic/claude-3-7-sonnet-latest"])
async def test_arun_instructor_with_payloads_pdf(model):
    """Test arun_instructor_with_payloads function with a single payload containing a PDF document."""
    # Set up test dependencies
    from adala.runtimes._litellm import InstructorAsyncClientMixin, AsyncRetrying
    from adala.utils.llm_utils import arun_instructor_with_payloads
    from adala.utils.parse import MessageChunkType
    
    class DocumentAnalysis(BaseModel):
        """Model for PDF document analysis response."""

        summary: str = Field(..., description="A summary of the document content")
        topics: List[str] = Field(..., description="Main topics covered in the document")
        document_type: str = Field(
            ..., description="The type of document (e.g., academic paper, report, manual)"
        )

    client = InstructorAsyncClientMixin(
        model=model,
        provider="openai" if 'openai' in model else "anthropic",
        api_key=os.getenv("OPENAI_API_KEY") if 'openai' in model else os.getenv("ANTHROPIC_API_KEY")
    ).client

    # Configure retry policy (3 attempts with 1 second delay)
    retries = AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
    )

    # Create sample PDF URL - this is a placeholder that will be replaced manually
    sample_pdf_url = "https://hs-sandbox-pub.s3.us-east-1.amazonaws.com/demo/4.uniform.pdf"

    # Check if model supports PDF input
    assert check_model_pdf_support(
        "openai/gpt-4o-mini"
    ), "Model should support PDF input"

    # Define user prompt template that includes the PDF
    user_prompt_template = "Please analyze this document and provide a summary: {document_url}"
    
    # Define input field types to specify the document_url is a PDF
    input_field_types = {
        "document_url": MessageChunkType.PDF_URL,
    }

    # Call the function being tested
    results = await arun_instructor_with_payloads(
        client=client,
        payloads=[{"document_url": sample_pdf_url}],  # Single item in the list
        user_prompt_template=user_prompt_template,
        response_model=DocumentAnalysis,
        model=model,
        canonical_model_provider_string=model,
        temperature=0.1,
        max_tokens=500,
        retries=retries,
        instructions_template="You are a helpful assistant that analyzes documents.",
        input_field_types=input_field_types,
        split_into_chunks=True
    )
    
    # Verify we got exactly one result
    assert len(results) == 1
    response = results[0]
    # Verify the response contains expected fields
    assert "summary" in response
    assert isinstance(response["summary"], str)
    assert len(response["summary"]) > 0
    assert "angina" in response["summary"].lower()
    
    # Verify topics are present and in expected format
    assert "topics" in response
    assert isinstance(response["topics"], list)
    assert len(response["topics"]) > 0
    assert any("angina" in topic.lower() for topic in response["topics"])
    assert all(isinstance(topic, str) for topic in response["topics"])
    
    # Verify document type
    assert "document_type" in response
    assert isinstance(response["document_type"], str)
    
    # Verify token usage and cost information is present with correct types
    assert "_prompt_tokens" in response
    assert isinstance(response["_prompt_tokens"], int)
    assert "_completion_tokens" in response
    assert isinstance(response["_completion_tokens"], int)
    assert "_prompt_cost_usd" in response
    assert isinstance(response["_prompt_cost_usd"], float)
    assert "_completion_cost_usd" in response
    assert isinstance(response["_completion_cost_usd"], float)
    assert "_total_cost_usd" in response
    assert isinstance(response["_total_cost_usd"], float)
    
    # Verify message counts with exact values
    assert "_message_counts" in response
    assert response["_message_counts"]["text"] == 2
    assert response["_message_counts"]["file"] == 1
    
    # Verify inference time is tracked
    assert "_inference_time" in response
    assert isinstance(response["_inference_time"], float)


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
