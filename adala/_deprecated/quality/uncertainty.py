import pandas as pd


def estimate_uncertainty(
    df: pd.DataFrame,
    instruction: str,
    prediction_column: str = 'predictions',
    output_column: str = 'uncertainty'
) -> pd.DataFrame:
    """
    Estimate uncertainty in a pandas DataFrame given LLM predictions
    """
    pass