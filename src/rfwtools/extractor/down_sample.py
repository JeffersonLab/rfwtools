"""This module provides down sampling-based feature extraction tools.

Typically, these will be used by DataSet.produce_feature_set().  However there is no reason why these can't be run
externally.

Basic Usage Example:
::

    from rfwtools.data_set import DataSet
    from rfwtools.extractor.downsample import down_sample_extractor

    # Setup a DataSet object and get some example data to work with
    ds = DataSet()
    ds.load_example_set_csv("my_example_set.csv")

    # Get a single example to work on
    ex = ds.example_set.loc[0, 'example']

    # Run on one example with defaults
    down_sample_extractor(ex)
    # Run on one example with only 2 signals being processed
    down_sample_extractor(ex, signals=['1_GMES', '1_PMES'])
    # Run on one example, but only include values before the fault on set.
    down_sample_extractor(ex, query="Time < 0")

    # Run this on every example in the example set and produce a corresponding feature set for pre-fault signal data.
    ds.produce_feature_set(down_sample_extractor, query="Time < 0")
"""
from typing import List

import pandas as pd
import numpy as np
import lttb

from .utils import get_example_data
from ..example import Example


def down_sample_extractor(example: Example, signals: List[str], step_size: int = 16, query: str = None) -> pd.DataFrame:
    """Standardize and down sample several signals and concatenate into a single row.

    Arguments:
        example:
            The example on which to operate
        signals:
            An explicit list of the example's columns to be down sampled (e.g., "1_GMES").
        step_size:
            This controls the down sampling behavior.  Only include the first sample out of every 'step_size' samples
        query:
            Argument passed to the ex.event_df to filter data prior to feature extraction, e.g. "Time <= 0".

    Returns:
         A DataFrame with a single row containing the the down sampled and concatenated signals.
    """

    # Get the data from the Example
    event_df = get_example_data(example, query)

    ds_signals = list()
    for i in signals:
        # Standardize the signal
        sig = event_df[i].values
        if np.std(sig) == 0:
            sig = sig - np.mean(sig)
        else:
            sig = (sig - np.mean(sig)) / np.std(sig)

        # Down sample the signal
        ds_signals.append(sig[::step_size])

    return pd.DataFrame(np.concatenate(ds_signals, axis=0)).T


def lttb_extractor(example: Example, signals: List[str], n_out: int, query: str = None) -> pd.DataFrame:
    """Extract features via lttb on individual signals from a set.  Loads/unloads data.

    LTTB is not a fixed time step method, but produces good graphical results.  It uses a Largest Triangle Three Bucket
    approach which picks points based on which would maximize the size of triangles created but points in adjacent
    buckets.

    Arguments:
        example: The example on which to operate
        signals: A list of the example's columns to be down sampled (e.g., "1_GMES").
        n_out: The number of points to be returned
        query: Argument passed to the ex.event_df to filter data prior to feature extraction, e.g. "Time <= 0".

    Returns:
        A DataFrame with a single row containing the the down sampled and concatenated signals.

    """

    # Get the data from the Example
    event_df = get_example_data(example, query)

    # Compute the lttp downsampling for each signal
    ds_signals = list()
    for i in signals:
        sig = event_df[i].values
        down_sampled = lttb.downsample(np.array([event_df.Time.values, sig]).T, n_out=n_out).T[1]
        ds_signals.append(down_sampled)

    return pd.DataFrame(np.concatenate(ds_signals, axis=0)).T


