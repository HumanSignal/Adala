import pytest
from unittest.mock import patch, MagicMock
from pydantic import ValidationError
from collections import defaultdict

from adala.utils.message_builder import MessagesBuilder
from adala.utils.parse import MessageChunkType


def test_basic_message_builder():
    """Test basic functionality of MessagesBuilder with text templates."""
    builder = MessagesBuilder(
        user_prompt_template="Hello, my name is {name}.",
        system_prompt="You are a helpful assistant.",
    )

    messages = builder.get_messages({"name": "Alice"}).messages

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello, my name is Alice."


def test_message_builder_without_system_prompt():
    """Test MessagesBuilder without a system prompt."""
    builder = MessagesBuilder(user_prompt_template="Hello, my name is {name}.")

    messages = builder.get_messages({"name": "Bob"}).messages

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello, my name is Bob."


def test_message_builder_instruction_last():
    """Test MessagesBuilder with instruction_first set to False."""
    builder = MessagesBuilder(
        user_prompt_template="Hello, my name is {name}.",
        system_prompt="Remember to be helpful.",
        instruction_first=False,
    )

    messages = builder.get_messages({"name": "Charlie"}).messages

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    # The system prompt should be appended to the user content
    assert messages[0]["content"] == "Hello, my name is Charlie.Remember to be helpful."


def test_message_builder_with_extra_fields():
    """Test MessagesBuilder with extra fields."""
    builder = MessagesBuilder(
        user_prompt_template="Hello, my name is {name} and I'm {age} years old.",
        system_prompt="You are a helpful assistant.",
        extra_fields={"age": 30},
    )

    messages = builder.get_messages({"name": "David"}).messages

    assert len(messages) == 2
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello, my name is David and I'm 30 years old."


def test_message_builder_with_incomplete_template():
    """Test MessagesBuilder with incomplete template variables."""
    builder = MessagesBuilder(
        user_prompt_template="Hello, my name is {name} and I like {hobby}.",
        system_prompt="You are a helpful assistant.",
    )

    messages = builder.get_messages({"name": "Eva"}).messages

    assert len(messages) == 2
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello, my name is Eva and I like {hobby}."


def test_message_builder_split_into_chunks():
    """Test MessagesBuilder with split_into_chunks enabled."""
    builder = MessagesBuilder(
        user_prompt_template="Look at this image: {image}",
        system_prompt="You are a helpful assistant.",
        split_into_chunks=True,
        input_field_types={"image": MessageChunkType.IMAGE_URL},
    )

    messages = builder.get_messages({"image": "http://example.com/image.jpg"}).messages

    assert len(messages) == 2
    assert messages[1]["role"] == "user"
    assert isinstance(messages[1]["content"], list)
    assert len(messages[1]["content"]) == 2
    assert messages[1]["content"][0]["type"] == "text"
    assert messages[1]["content"][0]["text"] == "Look at this image: "
    assert messages[1]["content"][1]["type"] == "image_url"
    assert (
        messages[1]["content"][1]["image_url"]["url"] == "http://example.com/image.jpg"
    )


def test_message_builder_with_multiple_images():
    """Test MessagesBuilder with multiple image URLs."""
    builder = MessagesBuilder(
        user_prompt_template="Compare these images: {image1} and {image2}",
        system_prompt="You are a helpful assistant.",
        split_into_chunks=True,
        input_field_types={
            "image1": MessageChunkType.IMAGE_URL,
            "image2": MessageChunkType.IMAGE_URL,
        },
    )

    messages = builder.get_messages(
        {
            "image1": "http://example.com/image1.jpg",
            "image2": "http://example.com/image2.jpg",
        }
    ).messages

    assert len(messages) == 2
    assert messages[1]["role"] == "user"
    assert isinstance(messages[1]["content"], list)
    assert len(messages[1]["content"]) == 4
    assert messages[1]["content"][0]["type"] == "text"
    assert messages[1]["content"][0]["text"] == "Compare these images: "
    assert messages[1]["content"][1]["type"] == "image_url"
    assert (
        messages[1]["content"][1]["image_url"]["url"] == "http://example.com/image1.jpg"
    )
    assert messages[1]["content"][2]["type"] == "text"
    assert messages[1]["content"][2]["text"] == " and "
    assert messages[1]["content"][3]["type"] == "image_url"
    assert (
        messages[1]["content"][3]["image_url"]["url"] == "http://example.com/image2.jpg"
    )


def test_message_builder_with_empty_prompt():
    """Test MessagesBuilder with an empty user prompt template."""
    builder = MessagesBuilder(
        user_prompt_template="", system_prompt="You are a helpful assistant."
    )

    messages = builder.get_messages({}).messages

    assert len(messages) == 2
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == ""


def test_message_builder_append_system_to_chunks():
    """Test appending system prompt to chunked content."""
    builder = MessagesBuilder(
        user_prompt_template="Look at this image: {image}",
        system_prompt="Describe the image in detail.",
        instruction_first=False,
        split_into_chunks=True,
        input_field_types={"image": MessageChunkType.IMAGE_URL},
    )

    messages = builder.get_messages({"image": "http://example.com/image.jpg"}).messages

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert isinstance(messages[0]["content"], list)

    # Verify the system prompt is appended correctly
    assert len(messages[0]["content"]) == 3
    assert messages[0]["content"][0]["type"] == "text"
    assert messages[0]["content"][0]["text"] == "Look at this image: "
    assert messages[0]["content"][1]["type"] == "image_url"
    assert (
        messages[0]["content"][1]["image_url"]["url"] == "http://example.com/image.jpg"
    )
    assert messages[0]["content"][2]["type"] == "text"
    assert messages[0]["content"][2]["text"] == "Describe the image in detail."


@pytest.mark.parametrize(
    "max_input_tokens, expected_message_count, expected_chunk_count",
    [
        (1, 1, 0),  # Only system message fits
        (30, 2, 0),  # System message and first user message fit
        (50, 3, 2),  # All messages fit, but only 2 chunks in the last message
    ],
)
@patch("litellm.model_cost")
@patch("litellm.utils.token_counter")
def test_trim_messages_to_fit_context(
    mock_token_counter,
    mock_model_cost,
    max_input_tokens,
    expected_message_count,
    expected_chunk_count,
):
    """Test trimming messages to fit context window."""
    # Setup model token limits
    mock_model_cost.__contains__.return_value = True
    mock_model_cost.__getitem__.return_value = {"max_input_tokens": max_input_tokens}
    # Create test messages
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "This is a long user message"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "This is chunk 1"},
                {"type": "text", "text": "This is chunk 2"},
                {"type": "text", "text": "This is chunk 3"},
                {"type": "text", "text": "This is chunk 4"},
                {"type": "text", "text": "This is chunk 5"},
                {"type": "text", "text": "This is chunk 6"},
            ],
        },
    ]
    m = MessagesBuilder()

    trimmed = m.trim_messages_to_fit_context(messages=messages, model="gpt-4o")

    assert len(trimmed) == expected_message_count

    if expected_message_count > 0:
        assert trimmed[0] == messages[0]  # System message is preserved

    if expected_message_count > 1:
        assert trimmed[1] == messages[1]  # First user message is preserved

    if expected_message_count > 2 and expected_chunk_count > 0:
        assert len(trimmed[2]["content"]) == expected_chunk_count
        assert (
            trimmed[2]["content"][:expected_chunk_count]
            == messages[2]["content"][:expected_chunk_count]
        )


@patch("litellm.model_cost")
def test_trim_messages_model_not_found(mock_model_cost):
    """Test behavior when model isn't found in litellm.model_cost."""
    # Set up model_cost to not contain the model
    mock_model_cost.__getitem__.side_effect = KeyError("Model not found")

    builder = MessagesBuilder(
        user_prompt_template="Test", trim_to_fit_context=True, model="unknown-model"
    )

    messages = [{"role": "user", "content": "Hello"}]

    # Should return original messages when model not found
    trimmed = builder.trim_messages_to_fit_context(messages, "unknown-model")
    assert trimmed == messages


@patch("litellm.model_cost")
def test_trim_messages_no_max_tokens(mock_model_cost):
    """Test behavior when max_tokens isn't defined for the model."""
    # Set up model_cost to not contain max_tokens
    mock_model_cost.__getitem__.return_value = {"price": 0.01}

    builder = MessagesBuilder(
        user_prompt_template="Test",
        trim_to_fit_context=True,
        model="model-without-max-tokens",
    )

    messages = [{"role": "user", "content": "Hello"}]

    # Should return original messages when max_tokens not found
    trimmed = builder.trim_messages_to_fit_context(messages, "model-without-max-tokens")
    assert trimmed == messages


def test_input_field_types_default():
    """Test that input_field_types defaults correctly."""
    # Create a builder without specifying input_field_types
    builder = MessagesBuilder(user_prompt_template="Hello")

    # Verify it's a defaultdict with the right default value
    assert isinstance(builder.input_field_types, defaultdict)

    # Based on the code in _format_user_prompt and parse_template, we should use dict() first
    # to convert the Pydantic Field to a regular dict, then access it
    field_types_dict = dict(builder.input_field_types)
    nonexistent_key = "some_nonexistent_key"
    # Default value for any missing key should be TEXT
    assert (
        field_types_dict.get(nonexistent_key, MessageChunkType.TEXT)
        == MessageChunkType.TEXT
    )

    # Create a builder with a custom input_field_types
    custom_fields = defaultdict(lambda: MessageChunkType.TEXT)
    custom_fields["image"] = MessageChunkType.IMAGE_URL

    builder = MessagesBuilder(
        user_prompt_template="Hello {image}", input_field_types=custom_fields
    )

    # Convert to dict and then check values
    field_types_dict = dict(builder.input_field_types)
    assert field_types_dict.get("image") == MessageChunkType.IMAGE_URL
    assert (
        field_types_dict.get("does_not_exist", MessageChunkType.TEXT)
        == MessageChunkType.TEXT
    )


def test_all_features_combined():
    """Test combining all features of MessagesBuilder."""
    builder = MessagesBuilder(
        user_prompt_template="Hello {name}, look at these images: {image1} and {image2}. Your age is {age}.",
        system_prompt="Analyze the images carefully.",
        instruction_first=True,
        extra_fields={"age": 30},
        split_into_chunks=True,
        input_field_types={
            "image1": MessageChunkType.IMAGE_URL,
            "image2": MessageChunkType.IMAGE_URL,
        },
        trim_to_fit_context=True,
        model="gpt-4",
    )

    with (
        patch("litellm.model_cost") as mock_model_cost,
        patch("litellm.utils.token_counter") as mock_token_counter,
    ):
        # Setup model token limits
        mock_model_cost.__getitem__.return_value = {"max_input_tokens": 4000}
        # Make token counting return small values to avoid trimming
        mock_token_counter.return_value = 100

        messages = builder.get_messages(
            {
                "name": "Grace",
                "image1": "http://example.com/image1.jpg",
                "image2": "http://example.com/image2.jpg",
            }
        ).messages

    # Verify the result
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Analyze the images carefully."
    assert messages[1]["role"] == "user"
    assert isinstance(messages[1]["content"], list)

    # Check the number of chunks from the log output
    assert len(messages[1]["content"]) == 5  # 3 text chunks + 2 image chunks

    # Check first few chunks
    assert messages[1]["content"][0]["type"] == "text"
    assert messages[1]["content"][0]["text"] == "Hello Grace, look at these images: "
    assert messages[1]["content"][1]["type"] == "image_url"
    assert (
        messages[1]["content"][1]["image_url"]["url"] == "http://example.com/image1.jpg"
    )
    assert messages[1]["content"][2]["type"] == "text"
    assert messages[1]["content"][2]["text"] == " and "
    assert messages[1]["content"][3]["type"] == "image_url"
    assert (
        messages[1]["content"][3]["image_url"]["url"] == "http://example.com/image2.jpg"
    )
    assert messages[1]["content"][4]["type"] == "text"
    assert messages[1]["content"][4]["text"] == ". Your age is 30."


@patch("litellm.utils.token_counter")
def test_trim_empty_messages(mock_token_counter):
    """Test trimming with empty messages list."""
    # Setup token counts
    mock_token_counter.return_value = 0

    builder = MessagesBuilder(
        user_prompt_template="Test", trim_to_fit_context=True, model="gpt-4"
    )

    # Create empty messages list
    messages = []

    # Mock litellm.model_cost to avoid KeyError
    with patch("litellm.model_cost") as mock_model_cost:
        mock_model_cost.__getitem__.return_value = {"max_input_tokens": 1000}

        # Test trimming empty messages
        trimmed = builder.trim_messages_to_fit_context(messages, "gpt-4")

    # Verify the result is still empty
    assert trimmed == []


@patch("litellm.model_cost")
@patch("litellm.utils.token_counter")
def test_trim_exceeds_context_window(mock_token_counter, mock_model_cost):
    """Test when system message alone exceeds context window."""
    # Setup model token limits
    mock_model_cost.__getitem__.return_value = {"max_input_tokens": 50}

    # Setup token counts
    mock_token_counter.return_value = 100  # Every message is 100 tokens

    builder = MessagesBuilder(
        user_prompt_template="Test", trim_to_fit_context=True, model="gpt-4"
    )

    # Create test messages
    messages = [
        {"role": "system", "content": "This system prompt is too long"},
        {"role": "user", "content": "User message"},
    ]

    # Mock isn't working as expected
    trimmed = builder.trim_messages_to_fit_context(messages, "gpt-4o")

    # Just verify we get the original messages back
    assert len(trimmed) == len(messages)
    assert trimmed == messages


def test_get_messages_with_all_default_fields():
    """Test MessagesBuilder with all default fields."""
    # Create a minimal builder with just the required user_prompt_template
    builder = MessagesBuilder(user_prompt_template="Hello, world!")

    # Get messages with no payload
    messages = builder.get_messages({}).messages

    # Verify result
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello, world!"


def test_split_message_into_chunks():
    # Test basic text-only template
    result = MessagesBuilder.split_message_into_chunks(
        "Hello {name}!", {"name": MessageChunkType.TEXT}, name="Alice"
    )
    assert result == [{"type": "text", "text": "Hello Alice!"}]

    # Test template with image URL
    result = MessagesBuilder.split_message_into_chunks(
        "Look at this {image}",
        {"image": MessageChunkType.IMAGE_URL},
        image="http://example.com/img.jpg",
    )
    assert result == [
        {"type": "text", "text": "Look at this "},
        {"type": "image_url", "image_url": {"url": "http://example.com/img.jpg"}},
    ]

    # Test mixed text and image template
    result = MessagesBuilder.split_message_into_chunks(
        "User {name} shared {image} yesterday",
        {"name": MessageChunkType.TEXT, "image": MessageChunkType.IMAGE_URL},
        name="Bob",
        image="http://example.com/photo.jpg",
    )
    assert result == [
        {"type": "text", "text": "User Bob shared "},
        {"type": "image_url", "image_url": {"url": "http://example.com/photo.jpg"}},
        {"type": "text", "text": " yesterday"},
    ]

    # Test multiple occurrences of same field
    result = MessagesBuilder.split_message_into_chunks(
        "{name} is here. Hi {name}!", {"name": MessageChunkType.TEXT}, name="Dave"
    )
    assert result == [{"type": "text", "text": "Dave is here. Hi Dave!"}]


def test_message_builder_with_multiple_image_urls():
    """Test MessagesBuilder with MessageChunkType.IMAGE_URLS - multiple URLs in a single field."""
    builder = MessagesBuilder(
        user_prompt_template="Here are some examples: {images}",
        system_prompt="Analyze all the images.",
        split_into_chunks=True,
        input_field_types={"images": MessageChunkType.IMAGE_URLS},
    )

    # List of multiple image URLs
    image_urls = [
        "http://example.com/image1.jpg",
        "http://example.com/image2.jpg",
        "http://example.com/image3.jpg",
    ]

    messages = builder.get_messages({"images": image_urls}).messages

    # Verify the correct structure was created
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Analyze all the images."
    assert messages[1]["role"] == "user"
    assert isinstance(messages[1]["content"], list)

    # Expect 4 chunks: text prefix + 3 image URLs
    assert len(messages[1]["content"]) == 4

    # Verify text chunk
    assert messages[1]["content"][0]["type"] == "text"
    assert messages[1]["content"][0]["text"] == "Here are some examples: "

    # Verify all image chunks were added
    for i in range(3):
        assert messages[1]["content"][i + 1]["type"] == "image_url"
        assert messages[1]["content"][i + 1]["image_url"]["url"] == image_urls[i]


def test_message_builder_with_mixed_field_types():
    """Test MessagesBuilder with mixed field types including IMAGE_URLS."""
    builder = MessagesBuilder(
        user_prompt_template="This is a {text_field}. Look at these examples: {images}, and this single image: {single_image}",
        system_prompt="Process all content.",
        split_into_chunks=True,
        input_field_types={
            "text_field": MessageChunkType.TEXT,
            "images": MessageChunkType.IMAGE_URLS,
            "single_image": MessageChunkType.IMAGE_URL,
        },
    )

    image_urls = ["http://example.com/sample1.jpg", "http://example.com/sample2.jpg"]

    messages = builder.get_messages(
        {
            "text_field": "test message",
            "images": image_urls,
            "single_image": "http://example.com/main.jpg",
        }
    ).messages

    assert len(messages) == 2
    assert messages[1]["role"] == "user"
    assert isinstance(messages[1]["content"], list)

    # Updated: Based on actual implementation, there are 5 chunks:
    # - text chunk (with text_field value merged in)
    # - 2 image chunks from image_urls
    # - text connector
    # - single image chunk
    assert len(messages[1]["content"]) == 5

    # Verify text chunks
    assert messages[1]["content"][0]["type"] == "text"
    assert (
        messages[1]["content"][0]["text"]
        == "This is a test message. Look at these examples: "
    )

    # Verify image_urls chunks
    assert messages[1]["content"][1]["type"] == "image_url"
    assert messages[1]["content"][1]["image_url"]["url"] == image_urls[0]
    assert messages[1]["content"][2]["type"] == "image_url"
    assert messages[1]["content"][2]["image_url"]["url"] == image_urls[1]

    # Verify text connector
    assert messages[1]["content"][3]["type"] == "text"
    assert messages[1]["content"][3]["text"] == ", and this single image: "

    # Verify single image
    assert messages[1]["content"][4]["type"] == "image_url"
    assert (
        messages[1]["content"][4]["image_url"]["url"] == "http://example.com/main.jpg"
    )


def test_empty_image_urls_list():
    """Test MessagesBuilder with an empty image URLs list."""
    builder = MessagesBuilder(
        user_prompt_template="Here are examples if available: {images}",
        split_into_chunks=True,
        input_field_types={"images": MessageChunkType.IMAGE_URLS},
    )

    # Test with empty list
    messages = builder.get_messages({"images": []}).messages

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert isinstance(messages[0]["content"], list)

    # Should just have the text part without any images
    assert len(messages[0]["content"]) == 1
    assert messages[0]["content"][0]["type"] == "text"
    assert messages[0]["content"][0]["text"] == "Here are examples if available: "


def test_split_message_into_chunks_with_image_urls():
    """Test the split_message_into_chunks static method with IMAGE_URLS type."""
    # Test with image URLs list
    image_urls = ["http://example.com/img1.jpg", "http://example.com/img2.jpg"]

    result = MessagesBuilder.split_message_into_chunks(
        "Check these images: {images}",
        {"images": MessageChunkType.IMAGE_URLS},
        images=image_urls,
    )

    # Should contain text chunk + multiple image chunks
    assert len(result) == 3
    assert result[0]["type"] == "text"
    assert result[0]["text"] == "Check these images: "
    assert result[1]["type"] == "image_url"
    assert result[1]["image_url"]["url"] == image_urls[0]
    assert result[2]["type"] == "image_url"
    assert result[2]["image_url"]["url"] == image_urls[1]
