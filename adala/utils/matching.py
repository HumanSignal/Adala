import pandas as pd
import difflib
from .internal_data import InternalSeries


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
