import pandas as pd
import time

from rich import print
from rich.table import Table
from rich import box
from rich.console import Console
from typing import Optional
from .internal_data import InternalDataFrame

console = Console()


def print_text(text: str, style=None, streaming_style=False):
    if streaming_style:
        for char in text:
            console.print(char, sep='', end='', style=style)
            time.sleep(0.01)
        console.print()
    else:
        console.print(text, style=style)


def print_dataframe(dataframe: InternalDataFrame):
    num_rows = 5
    table = Table(show_header=True, header_style="bold magenta")
    # index_name = dataframe.index.name or 'index'
    # table.add_column(index_name)

    for column in dataframe.columns:
        table.add_column(str(column))

    for index, value_list in enumerate(dataframe.iloc[:num_rows].values.tolist()):
        # row = [str(index)]
        row = []
        row += [str(x) for x in value_list]
        table.add_row(*row)

    # Update the style of the table
    table.row_styles = ["none", "dim"]
    table.box = box.SIMPLE_HEAD

    console.print(table)
