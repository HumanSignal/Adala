import pandas as pd


def human_in_the_loop(
    df: pd.DataFrame,
    label_studio_project_id: int,
    output_column: str = 'predictions'
) -> pd.DataFrame:
    """
    Auto-annotate a pandas DataFrame with human-in-the-loop labeling from Label Studio.
    """
    pass