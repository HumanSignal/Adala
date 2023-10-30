import pandas as pd
import difflib
from .internal_data import InternalSeries


# Function to apply fuzzy matching
def _fuzzy_match(str1, str2, match_threshold=0.95):
    ratio = difflib.SequenceMatcher(None, str1.strip(), str2.strip()).ratio()
    return ratio >= match_threshold


def fuzzy_match(x: InternalSeries, y: InternalSeries, threshold=0.8):
    result = x.combine(y, lambda x, y: _fuzzy_match(x, y, threshold))
    return result
