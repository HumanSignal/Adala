import pandas as pd
import textwrap as tw


def log(message: str):
    print(message)


def info(message: str = ''):
    print(message)


def print_instructions(text: str, compact: bool = False):
    segments = text.split('\n')
    wrapped_segments = [
        tw.fill(segment, width=100, initial_indent=" " * 4, subsequent_indent=" " * 4)
        for segment in segments
    ]
    wrapped_text = '\n'.join(wrapped_segments)
    if compact:
        wrapped_text = wrapped_text.replace('\n\n', '\n')

    print(f'Instructions = \n{wrapped_text}')


def print_evaluations(evaluations: pd.DataFrame):
    evaluations = evaluations[evaluations.columns.difference(['score', 'ground_truth__x__sentiment'])[::-1]]
    lines = str(evaluations)
    lines = '    ' + '\n    '.join(lines.split('\n'))
    print(lines)