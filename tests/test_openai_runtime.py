from .utils import patching, PatchedCalls, mdict


@patching(
    target_function=PatchedCalls.OPENAI_MODEL_LIST.value,
    data=[],
)
@patching(
    target_function=PatchedCalls.GUIDANCE.value,
    data=[
        {
            "input": dict(
                text_="Hello", comments="Yes", silent=True, label_options=list("abc")
            ),
            "output": {"output": "Hello, World!", "label": "World"},
        },
        {
            "input": dict(
                text_="Test", comments="No", silent=True, label_options=list("abc")
            ),
            "output": {"output": "Hello, Test!", "label": "Test"},
        },
    ],
    strict=False,
)
def test_process_batch():
    from adala.utils.internal_data import InternalDataFrame  # type: ignore
    from adala.runtimes import GuidanceRuntime  # type: ignore

    df = InternalDataFrame(
        [["Hello", "Yes"], ["Test", "No"]], columns=["text", "comments"]
    )

    runtime = GuidanceRuntime()
    result = runtime.batch_to_batch(
        batch=df,
        input_template="Input: {text} {comments}",
        output_template="Output: {output} {label}",
        instructions_template="This is a test.",
        field_schema={"label": {"type": "array", "items": {"enum": list("abc")}}},
    )
    assert isinstance(result, InternalDataFrame)
    assert result.equals(
        InternalDataFrame(
            [["Hello, World!", "World"], ["Hello, Test!", "Test"]],
            columns=["output", "label"],
        )
    )
