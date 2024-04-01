import pandas as pd

from utils import patching, PatchedCalls, OpenaiChatCompletionMock, mdict
from openai.types import CreateEmbeddingResponse, Embedding

@patching(
    target_function=PatchedCalls.OPENAI_MODEL_RETRIEVE.value,
    data=[
        # calling API model list for the student runtime
        {
            "input": {},
            "output": {},
        },
        # calling API model list for the teacher runtime
        {
            "input": {},
            "output": {},
        },
    ],
)
@patching(
    target_function=PatchedCalls.OPENAI_CHAT_COMPLETION.value,
    data=[
        # Forward pass for the student runtime
        {
            "input": {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "system",
                        "content": "Recognize emotions from text.\n\nAssume the following output labels:\n\nhappy\nsad\nangry\nneutral\n\nDon't output anything else - only respond with one of the labels above.",
                    },
                    {
                        "role": "user",
                        "content": "Examples:\n\n\n\nNow recognize:\n\nText: I am happy\nEmotions: ",
                    },
                ],
            },
            "output": OpenaiChatCompletionMock("happy"),
        },
        {
            "input": {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "system",
                        "content": "Recognize emotions from text.\n\nAssume the following output labels:\n\nhappy\nsad\nangry\nneutral\n\nDon't output anything else - only respond with one of the labels above.",
                    },
                    {
                        "role": "user",
                        "content": "Examples:\n\n\n\nNow recognize:\n\nText: I am angry\nEmotions: ",
                    },
                ],
            },
            "output": OpenaiChatCompletionMock("sad"),  # wrong prediction
        },
        {
            "input": {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "system",
                        "content": "Recognize emotions from text.\n\nAssume the following output labels:\n\nhappy\nsad\nangry\nneutral\n\nDon't output anything else - only respond with one of the labels above.",
                    },
                    {
                        "role": "user",
                        "content": "Examples:\n\n\n\nNow recognize:\n\nText: I am sad\nEmotions: ",
                    },
                ],
            },
            "output": OpenaiChatCompletionMock("neutral"),  # wrong prediction
        },
        # Backward pass for the student runtime
        {
            "input": {"model": "gpt-4"},
            "output": OpenaiChatCompletionMock("Reasoning..."),
        },
        {
            "input": {"model": "gpt-4"},
            "output": OpenaiChatCompletionMock("IMPROVED INSTRUCTION!"),
        },
        # Forward pass for the student runtime, 2nd iteration, RAG is now updated
        {
            "input": {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "IMPROVED INSTRUCTION!"},
                    {
                        "role": "user",
                        "content": """\
Examples:

Text: I am angry
Emotions: angry

Text: I am sad
Emotions: sad

Now recognize:

Text: I am happy
Emotions: """,
                    },
                ],
            },
            "output": OpenaiChatCompletionMock("happy"),
        },
        {
            "input": {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "IMPROVED INSTRUCTION!"},
                    {
                        "role": "user",
                        "content": """\
Examples:

Text: I am angry
Emotions: angry

Text: I am sad
Emotions: sad

Now recognize:

Text: I am angry
Emotions: """,
                    },
                ],
            },
            "output": OpenaiChatCompletionMock(content="angry"),
        },
        # notice that the RAG skill must move "sad" to the top of the list, compare to the previous example with "angry" at the top
        {
            "input": {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "IMPROVED INSTRUCTION!"},
                    {
                        "role": "user",
                        "content": """\
Examples:

Text: I am sad
Emotions: sad

Text: I am angry
Emotions: angry

Now recognize:

Text: I am sad
Emotions: """,
                    },
                ],
            },
            "output": OpenaiChatCompletionMock("sad"),
        },
    ],
    strict=False,
)
@patching(
    target_function=PatchedCalls.OPENAI_EMBEDDING_CREATE.value,
    data=[
        # calculate embeddings on forward pass for the student runtime
        {
            "input": {
                "model": "text-embedding-ada-002",
                "input": [
                    "Text: I am happy",
                    "Text: I am angry",
                    "Text: I am sad"
                ],
            },
            "output": CreateEmbeddingResponse(
                data=[
                    Embedding(index=0, embedding=[1, 0, 0], object="embedding"),
                    Embedding(index=1, embedding=[1, 1, 0], object="embedding"),
                    # assume angry is closer to happy than sad
                    Embedding(index=2, embedding=[0, 0, 1], object="embedding"),
                ],
                model="text-embedding-ada-002",
                object="list",
                usage={'prompt_tokens': 0, 'total_tokens': 0}
            )
        },
        # calculate embedding on error example while improving RAG skill
        {
            "input": {
                "model": "text-embedding-ada-002",
                "input": [
                    "Text: I am angry",  # error example
                    "Text: I am sad",  # error example
                ],
            },
            "output":
                CreateEmbeddingResponse(
                    data=[
                        Embedding(index=1, embedding=[1, 1, 0], object="embedding"),
                        # assume angry is closer to happy than sad
                        Embedding(index=2, embedding=[0, 0, 1], object="embedding"),
                    ],
                    model="text-embedding-ada-002",
                    object="list",
                    usage={'prompt_tokens': 0, 'total_tokens': 0}
                ),
        },
        # calculate embeddings on 2nd forward pass for the student runtime
        {
            "input": {
                "model": "text-embedding-ada-002",
                "input": [
                    "Text: I am happy",
                    "Text: I am angry",
                    "Text: I am sad"
                ],
            },
            "output": CreateEmbeddingResponse(
                data=[
                    Embedding(index=0, embedding=[1, 0, 0], object="embedding"),
                    Embedding(index=1, embedding=[1, 1, 0], object="embedding"),
                    # assume angry is closer to happy than sad
                    Embedding(index=2, embedding=[0, 0, 1], object="embedding"),
                ],
                model="text-embedding-ada-002",
                object="list",
                usage={'prompt_tokens': 0, 'total_tokens': 0}
            )
        },
    ],
)
def test_rag_with_openai_chat_completion():
    from adala.agents import Agent  # type: ignore
    from adala.skills import LinearSkillSet, ClassificationSkill, RAGSkill  # type: ignore
    from adala.environments import StaticEnvironment  # type: ignore
    from adala.runtimes import OpenAIChatRuntime  # type: ignore
    from adala.memories import VectorDBMemory  # type: ignore

    memory = VectorDBMemory(db_name="emotions", openai_api_key='abcde')

    agent = Agent(
        skills=LinearSkillSet(
            skills=[
                RAGSkill(
                    name="rag",
                    input_template="Text: {text}",
                    output_template="{examples}",
                    rag_input_template="Text: {text}\nEmotions: {emotions}",
                    num_results=2,
                    memory=memory,
                ),
                ClassificationSkill(
                    name="emotions",
                    instructions="Recognize emotions from text.",
                    input_template="Examples:\n\n{examples}\n\nNow recognize:\n\nText: {text}",
                    output_template="Emotions: {prediction}",
                    labels={"prediction": ["happy", "sad", "angry", "neutral"]},
                ),
            ]
        ),
        environment=StaticEnvironment(
            ground_truth_columns={"prediction": "emotions"},
            df=pd.DataFrame(
                {
                    "text": [
                        "I am happy",
                        "I am angry",
                        "I am sad",
                    ],
                    "emotions": [
                        "happy",
                        "angry",
                        "sad",
                    ],
                }
            ),
        ),
        runtimes={"default": OpenAIChatRuntime(model="gpt-3.5-turbo")},
        teacher_runtimes={"openai-teacher": OpenAIChatRuntime(model="gpt-4")},
        default_teacher_runtime="openai-teacher",
    )
    agent.learn(learning_iterations=2)
