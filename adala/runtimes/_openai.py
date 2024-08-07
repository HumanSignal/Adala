from ._litellm import AsyncLiteLLMChatRuntime, LiteLLMChatRuntime, LiteLLMVisionRuntime


# litellm already reads the OPENAI_API_KEY env var, which was the reason for this class
OpenAIChatRuntime = LiteLLMChatRuntime
AsyncOpenAIChatRuntime = AsyncLiteLLMChatRuntime
OpenAIVisionRuntime = LiteLLMVisionRuntime
