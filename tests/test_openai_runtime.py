from utils import patching, PatchedCalls


@patching(
    target_function=PatchedCalls.GUIDANCE.value,
    data=[{
        'input': dict(text_='Hello', comments='Yes', silent=True, labels=list('abc')),
        'output': {'output': 'Hello, World!', 'label': 'World', 'logprobs': {'World': -0.1, 'Test': -0.2}}
    }, {
        'input': dict(text_='Test', comments='No', silent=True, labels=list('abc')),
        'output': {'output': 'Hello, Test!', 'label': 'Test', 'logprobs': {'World': 0.2, 'Test': 0.1}}
    }],
    strict=False
)
def test_process_batch():
    from adala.utils.internal_data import InternalDataFrame
    from adala.runtimes.openai import OpenAIRuntime

    df = InternalDataFrame(
        [['Hello', 'Yes'],
        ['Test', 'No']],
        columns=['text', 'comments']
    )

    runtime = OpenAIRuntime()
    result = runtime.process_batch(
        batch=df,
        input_template='Input: {{text}} {{comments}}',
        output_template="Output: {{gen 'output'}} {{select 'label' options=labels logprobs='logprobs'}}",
        instructions='This is a test.',
        extra_fields={'labels': list('abc')}
    )
    assert isinstance(result, InternalDataFrame)
    assert result.equals(InternalDataFrame([
        ['Hello, World!', 'World', {'World': -0.1, 'Test': -0.2}],
        ['Hello, Test!', 'Test', {'World': 0.2, 'Test': 0.1}]
    ], columns=['output', 'label', 'logprobs']))
