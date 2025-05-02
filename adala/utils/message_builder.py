"""This module provides a utility class for building and formatting messages for LLM interactions.
The main interface includes:
- MessagesBuilder.get_messages() -> MessagesBuilderGetMessagesResponse

Other methods are used internally and not intended for external use.
"""

import logging
from typing import Dict, Any, Optional, List, DefaultDict, Annotated, Union
from collections import defaultdict

from pydantic import BaseModel, Field, field_validator
from pydantic.dataclasses import dataclass

from adala.utils.token_counter import TokenCounter, get_token_counter
from adala.utils.parse import (
    MessageChunkType,
    MessageChunk,
    parse_template,
    partial_str_format,
)

logger = logging.getLogger(__name__)


@dataclass
class MessagesBuilderGetMessagesResponse:
    """
    Response model for MessagesBuilder.get_messages method.

    Contains the formatted messages ready to be sent to an LLM API.
    """

    messages: List[Dict[str, Any]] = Field(
        description="List of message dictionaries formatted for LLM API consumption"
    )


class MessagesBuilder(BaseModel):
    """
    A utility class for building and formatting messages for LLM interactions.

    This class handles the construction of message payloads for LLM APIs, supporting
    various formatting options including text and image content, system prompts,
    and context management.

    Attributes:
        user_prompt_template (str): Template string for formatting user messages.
        system_prompt (Optional[str]): Optional system prompt to include in messages.
        instruction_first (bool): Whether to place instructions before user content.
        extra_fields (Dict[str, Any]): Additional fields to include in messages.
        split_into_chunks (bool): Whether to split messages into chunks by content type.
            For example, if the user prompt contains an image URL ("Analyze this image: {image}"), it will be split into
            a text chunk and an image chunk formatted as:
            [
                {"type": "text", "text": "Analyze this image: "},
                {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}},
            ]
        input_field_types (DefaultDict[str, MessageChunkType]): Maps field names to their content types.
            Available types:
            - MessageChunkType.TEXT: Text content
            - MessageChunkType.IMAGE_URL: Image URL
            - MessageChunkType.IMAGE_URLS: List of image URLs
            - MessageChunkType.PDF_URL: PDF document URL
        trim_to_fit_context (bool): Whether to trim messages to fit within model context limits.
            If True, the messages will be trimmed to fit within the model's context window (e.g. 128k tokens for GPT-4o).
            Warning: this will slow down the function.
        model (Optional[str]): The LLM model identifier, used for context window calculations.
            If not provided, the model will not be trimmed.
        token_counter (TokenCounter): The token counter to use for context window calculations.
            If not provided, the default token counter will be used.

    Example:
        ```python
        builder = MessagesBuilder(
            user_prompt_template="Analyze this image: {image}",
            system_prompt="You are a helpful assistant.",
            split_into_chunks=True,
            input_field_types={"image": MessageChunkType.IMAGE_URL}
        )

        r = builder.get_messages({"image": "http://example.com/image.jpg"})
        print(r.messages)
        ```
        Output:
        ```
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}}]}
        ]
        ```

        Example with PDF:
        ```python
        builder = MessagesBuilder(
            user_prompt_template="Analyze this document: {document}",
            split_into_chunks=True,
            input_field_types={"document": MessageChunkType.PDF_URL}
        )

        r = builder.get_messages({"document": "http://example.com/document.pdf"})
        print(r.messages)
        ```
        Output:
        ```
        [
            {"role": "user", "content": [
                {"type": "text", "text": "Analyze this document: "},
                {"type": "file", "file": {
                    "file_id": "http://example.com/document.pdf",
                    "format": "application/pdf"
                }}]
            }
        ]
        ```
    """

    user_prompt_template: str = Field(default="")
    system_prompt: Optional[str] = Field(default=None)
    instruction_first: bool = Field(default=True)
    extra_fields: Dict[str, Any] = Field(default_factory=dict)
    split_into_chunks: bool = Field(default=False)
    input_field_types: Optional[
        DefaultDict[
            str,
            Annotated[
                MessageChunkType, Field(default_factory=lambda: MessageChunkType.TEXT)
            ],
        ]
    ] = Field(default_factory=lambda: defaultdict(lambda: MessageChunkType.TEXT))
    trim_to_fit_context: bool = Field(default=False)
    model: Optional[str] = Field(default=None)

    # compositional dependencies
    token_counter: TokenCounter = Field(default_factory=get_token_counter)

    class Config:
        arbitrary_types_allowed = True

    @field_validator("input_field_types", mode="before")
    @classmethod
    def set_default_input_field_types(cls, value):
        """Set default input field types if not provided."""
        if value is None:
            return defaultdict(lambda: MessageChunkType.TEXT)
        return value

    @classmethod
    def split_message_into_chunks(
        cls,
        input_template: str,
        input_field_types: Dict[str, MessageChunkType],
        **payload,
    ) -> List[MessageChunk]:
        """Split a template string into message chunks based on field types.

        Args:
            input_template: Template string with placeholders like '{field_name}'
            input_field_types: Mapping of field names to their chunk types
            payload: Dictionary with values to substitute into the template instead of placeholders

        Returns:
            List of message chunks with appropriate type and content:
            - Text chunks: {'type': 'text', 'text': str}
            - Image chunks: {'type': 'image_url', 'image_url': {'url': str}}

        Example:
            >>> split_message_into_chunks(
            ...     'Look at {image} and describe {text}',
            ...     {'image': MessageChunkType.IMAGE_URL, 'text': MessageChunkType.TEXT},
            ...     {'image': 'http://example.com/img.jpg', 'text': 'this content'}
            ... )
            [
                {'type': 'text', 'text': 'Look at '},
                {'type': 'image_url', 'image_url': {'url': 'http://example.com/img.jpg'}},
                {'type': 'text', 'text': ' and describe this content'}
            ]
        """
        # Parse template to get chunks with field positions and types
        parsed = parse_template(
            input_template,
            include_texts=True,
            payload=payload,
            input_field_types=input_field_types,
        )

        logger.debug("Parsed template: %s", parsed)

        result = []
        current_text = ""

        def _add_current_text_as_chunk():
            # this function is used to flush `current_text` buffer into a text chunk, and start over
            nonlocal current_text
            if current_text:
                result.append({"type": "text", "text": current_text})
                current_text = ""

        for part in parsed:
            # iterate over parsed chunks - they already contains field types and placeholder values
            if part["type"] == "text":
                # each text chunk without placeholders is added to the current buffer and we continue
                current_text += part["text"]
            elif part["type"] == "var":
                field_type = part["field_type"]
                field_value = part["data"]
                if field_value is None:
                    # if field value is not provided, it is assumed to be a text field
                    current_text += part["text"]
                else:
                    match field_type:
                        case MessageChunkType.TEXT:
                            # For text fields, we don't break chunks and add text fields to current buffer
                            current_text += (
                                str(field_value)
                                if field_value is not None
                                else part["text"]
                            )

                        case MessageChunkType.IMAGE_URL:
                            # Add remaining text as text chunk
                            _add_current_text_as_chunk()
                            # Add image URL as new image chunk
                            result.append(
                                {"type": "image_url", "image_url": {"url": field_value}}
                            )

                        case MessageChunkType.PDF_URL:
                            # Add remaining text as text chunk
                            _add_current_text_as_chunk()
                            # Add PDF URL as new file chunk
                            result.append(
                                {
                                    "type": "file",
                                    "file": {
                                        "file_id": field_value,
                                        "format": "application/pdf",
                                    },
                                }
                            )

                        case MessageChunkType.IMAGE_URLS:
                            assert isinstance(
                                field_value, List
                            ), "Image URLs must be a list"
                            # Add remaining text as text chunk
                            _add_current_text_as_chunk()
                            # Add image URLs as new image chunks
                            for url in field_value:
                                result.append(
                                    {"type": "image_url", "image_url": {"url": url}}
                                )

                        case _:
                            # Handle unknown field types as text
                            current_text += part["text"]

        # Add any remaining text
        _add_current_text_as_chunk()

        logger.debug("Result: %s", result)

        return result

    def get_messages(
        self, payload: Dict[str, Any]
    ) -> MessagesBuilderGetMessagesResponse:
        """
        Generate formatted messages based on template and payload.

        Args:
            payload: Variables to substitute in the template

        Example:
            >>> builder = MessagesBuilder(
            ...     user_prompt_template="Analyze this image: {image}",
            ...     system_prompt="You are a helpful assistant.",
            ...     split_into_chunks=True,
            ...     input_field_types={"image": MessageChunkType.IMAGE_URL}
            ... )
            >>> messages = builder.get_messages({"image": "http://example.com/image.jpg"})
            >>> print(messages)
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Analyze this image: "},
                    {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}}
                ]}
            ]

        Returns:
            List of message dictionaries ready for the LLM
        """
        # Format user prompt
        user_prompt = self._format_user_prompt(payload)

        # Construct messages list
        messages = self._build_messages_list(user_prompt)

        # Apply trimming if needed
        if self.trim_to_fit_context and self.model:
            messages = self.trim_messages_to_fit_context(messages, self.model)

        return MessagesBuilderGetMessagesResponse(messages=messages)

    def _format_user_prompt(
        self, payload: Dict[str, Any]
    ) -> Union[str, List[MessageChunk]]:
        """Format the user prompt using the template and payload."""
        if self.split_into_chunks:
            extra_fields_dict = dict(self.extra_fields) if self.extra_fields else {}
            combined_payload = {**payload, **extra_fields_dict}
            return self.split_message_into_chunks(
                input_template=self.user_prompt_template,
                input_field_types=(
                    dict(self.input_field_types) if self.input_field_types else {}
                ),
                **combined_payload,
            )
        else:
            extra_fields_dict = dict(self.extra_fields) if self.extra_fields else {}
            return partial_str_format(
                self.user_prompt_template, **payload, **extra_fields_dict
            )

    def _build_messages_list(
        self, user_prompt: Union[str, List[MessageChunk]]
    ) -> List[Dict[str, Any]]:
        """Build the messages list from the formatted user prompt and system prompt."""
        messages = [{"role": "user", "content": user_prompt}]

        if not self.system_prompt:
            return messages

        if self.instruction_first:
            # Add system message as first message
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        else:
            # Append system message to user content
            if isinstance(user_prompt, list) and isinstance(
                messages[0]["content"], list
            ):
                # For chunked content - create a typed message chunk for the system prompt
                system_msg: MessageChunk = {"type": "text", "text": self.system_prompt}
                messages[0]["content"] = messages[0]["content"] + [system_msg]
            else:
                # For string content
                if isinstance(messages[0]["content"], str):
                    messages[0]["content"] += self.system_prompt
                else:
                    # Handle the case where content is a list but not as expected
                    messages.insert(
                        0, {"role": "system", "content": self.system_prompt}
                    )

        return messages

    def trim_messages_to_fit_context(
        self, messages: List[Dict[str, Any]], model: str
    ) -> List[Dict[str, Any]]:
        """
        Trims messages to fit within the model's context window.

        Args:
            messages: List of messages to trim.
            model: The model name to use for token counting.

        Returns:
            List of trimmed messages that fit within the context window.
        """
        try:
            # Get token limit for the model
            max_tokens = self.token_counter.max_tokens(model)
            if max_tokens is None:
                logger.error(
                    "Could not determine max_tokens for model '%s'. Cannot trim.",
                    model,
                )
                return messages
        except Exception as e:
            logger.error(
                "Error accessing token limits for model '%s': %s. Cannot trim.",
                model,
                e,
            )
            return messages

        result_messages = []
        total_tokens = 0

        # Handle system message specially to preserve it if possible
        system_message = None
        if messages and messages[0]["role"] == "system":
            system_message = messages[0]
            system_tokens = self.token_counter.count_tokens(
                model=model, messages=[system_message]
            )
            total_tokens += system_tokens
            messages_to_process = messages[1:]
        else:
            messages_to_process = messages

        # Process remaining messages
        for message in messages_to_process:
            role = message["role"]
            content = message["content"]

            if isinstance(content, str):
                # Simple string content
                message_tokens = self.token_counter.count_tokens(
                    model=model, messages=[message]
                )
                if total_tokens + message_tokens <= max_tokens:
                    result_messages.append(message)
                    total_tokens += message_tokens
                else:
                    break

            elif isinstance(content, list):

                # List of content chunks
                new_content = []

                for chunk in content:
                    temp_message = {"role": role, "content": [chunk]}
                    chunk_tokens = self.token_counter.count_tokens(
                        model=model, messages=[temp_message]
                    )

                    if total_tokens + chunk_tokens <= max_tokens:
                        new_content.append(chunk)
                        total_tokens += chunk_tokens
                    else:
                        break

                if new_content:
                    result_messages.append({"role": role, "content": new_content})

        # Add system message if we have room
        if system_message:
            if result_messages:
                result_messages.insert(0, system_message)
            else:
                result_messages.append(system_message)

        if len(result_messages) != len(messages):
            logger.debug(
                "Trimmed messages from %d to %d", len(messages), len(result_messages)
            )

        return result_messages
