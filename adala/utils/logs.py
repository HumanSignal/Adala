import pandas as pd
import time

from rich import print
from rich.table import Table
from rich import box
from rich.console import Console
from typing import Optional
from .internal_data import InternalDataFrame, InternalSeries
from difflib import ndiff

console = Console()
error_console = Console(stderr=True, style="bold red")


def print_text(text: str, style=None, streaming_style=False):
    """
    Print text to console with optional style and streaming style.
    """
    if streaming_style:
        for char in text:
            console.print(char, sep="", end="", style=style)
            time.sleep(0.01)
        console.print()
    else:
        console.print(text, style=style)


def print_error(text: str):
    """
    Print error message to console.
    """
    error_console.print(text)


def print_dataframe(dataframe: InternalDataFrame):
    """
    Print dataframe to console.
    """
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


def print_series(data: InternalSeries):
    """
    Print series to console.
    """

    # Create a Rich Table with a column for each series value
    table = Table(show_header=True, header_style="bold magenta")

    # Add a column for each value in the series with the index as the header
    for index in data.index:
        table.add_column(str(index))

    # Add a single row with all the values from the series
    table.add_row(*[str(value) for value in data])

    # Print the table with the Rich console
    console.print(table)


def is_running_in_jupyter():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:
            return False
        return True
    except (AttributeError, ImportError):
        return False


def highlight_differences(text1, text2):
    from IPython.display import HTML, display

    diff = ndiff(text1, text2)
    highlighted = "".join(
        [
            (
                '<span style="background-color: lightgreen;">' + i[2] + "</span>"
                if i[0] == "+"
                else i[2]
            )
            for i in diff
            if i[0] != "-"
        ]
    )
    highlighted = highlighted.replace(" \n ", "<br>")
    display(HTML(highlighted))
