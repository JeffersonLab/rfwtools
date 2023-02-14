"""Utility functions to help with feature extraction."""

from rfwtools.example import Example
import pandas as pd


def get_example_data(ex: Example, query: str) -> pd.DataFrame:
    """Get the requested data from the provided Example.  Handles loading and unloading of Example data.

    Arguments:
        ex: The Example object
        query: The query parameter to apply to the Example.event_df or None if no query is requested

    Returns:
         The requested Example's event_df DataFrame
    """
    # Get the data
    ex.load_data()
    event_df = ex.event_df.copy()
    ex.unload_data()

    # Apply query if requested
    if query is not None:
        event_df = event_df.query(query)

    return event_df
