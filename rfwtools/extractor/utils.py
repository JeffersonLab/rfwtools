def get_example_data(ex, query):
    """Get the requested data from the provided Example.

    Args:
        ex (Example) - The Example object
        query (str) - The query parameter to apply to the Example.event_df or None if no query is requested

    Returns (DataFrame) - The requested Example's data
    """
    # Get the data
    ex.load_data()
    event_df = ex.event_df.copy()
    ex.unload_data()

    # Apply query if requested
    if query is not None:
        event_df = event_df.query(query)

    return event_df
