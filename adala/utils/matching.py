import pandas as pd
import difflib
from .internal_data import InternalSeries
from typing import List, Optional


# Function to apply fuzzy matching
def _fuzzy_match(str1, str2, match_threshold=0.95):
    ratio = difflib.SequenceMatcher(None, str(str1).strip(), str(str2).strip()).ratio()
    return ratio >= match_threshold


def fuzzy_match(x: InternalSeries, y: InternalSeries, threshold=0.8) -> InternalSeries:
    """
    Fuzzy match string values in two series.

    Args:
        x (InternalSeries): The first series.
        y (InternalSeries): The second series.
        threshold (float): The threshold to use for fuzzy matching. Defaults to 0.8.

    Returns:
        InternalSeries: The series with fuzzy match results.
    """
    result = x.combine(y, lambda x, y: _fuzzy_match(x, y, threshold))
    return result


def match_options(query: str, options: List[str], splitter: str = None) -> str:
    """
    Match a query to a list of options.
    If splitter is not None, the query will be split by the splitter and each part will be matched separately, then joined by the splitter.

    Args:
        query (str): The query.
        options (List[str]): The options.
        splitter (str): The splitter. Defaults to None.

    Returns:
        str: The matched option.
    """

    # hard constraint: the item must be in the query
    filtered_items = [item for item in options if item in query]
    if not filtered_items:
        # make the best guess - find the most similar item to the query
        filtered_items = options

    # soft constraint: find the most similar item to the query
    matched_items = []
    # split query by self.splitter
    if splitter:
        qs = query.split(splitter)
    else:
        qs = [query]

    for q in qs:
        scores = list(
            map(
                lambda item: difflib.SequenceMatcher(None, q, item).ratio(),
                filtered_items,
            )
        )
        matched_items.append(filtered_items[scores.index(max(scores))])
    if splitter:
        return splitter.join(matched_items)
    return matched_items[0]
