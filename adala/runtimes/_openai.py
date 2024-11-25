from ._litellm import AsyncLiteLLMChatRuntime, LiteLLMChatRuntime, AsyncLiteLLMVisionRuntime 


# litellm already reads the OPENAI_API_KEY env var, which was the reason for this class
OpenAIChatRuntime = LiteLLMChatRuntime
AsyncOpenAIChatRuntime = AsyncLiteLLMChatRuntime
AsyncOpenAIVisionRuntime = AsyncLiteLLMVisionRuntime