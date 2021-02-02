import pandas as pd
import numpy as np
import lttb

from .utils import get_example_data


def down_sample_extractor(example, signals, step_size=16, query=None):
    """Standardize and down sample several signals and concatenate into a single row.

    Args:
        example (Example) - The example on which to operate
        signals (list(str)) - An explicit list of the example's columns to be down sampled (e.g., "1_GMES").
        step_size (int) - This controls the down sampling behavior.  Only include the first sample out of every
                         'step_size' samples
        query (str) - Argument passed to the ex.event_df to filter data prior to feature extraction, e.g. "Time <= 0".

    Returns A DataFrame with a single row containing the the down sampled and concatenated signals.
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


def lttb_extractor(example, signals, n_out, query=None):
    """Extract features via lttb on individual signals from a set.  Loads/unloads data.

    LTTB is not a fixed time step method, but produces good graphical results.  It uses a Largest Triangle Three Bucket
    approach which picks points based on which would maximize the size of triangles created but points in adjacent
    buckets.

    Args:
        example (Example) - The example on which to operate
        signals (list(str)) - A list of the example's columns to be down sampled (e.g., "1_GMES").
        n_out (numeric) - The number of points to be returned
        query (str) - Argument passed to the ex.event_df to filter data prior to feature extraction, e.g. "Time <= 0".
    """

    # Get the data from the Example
    event_df = get_example_data(example, query)

    # Compute the lttp downsampling for each signal
    ds_signals = list()
    for i in signals:
        sig = event_df[i].values
        down_sampled = lttb.downsample(np.array([event_df.Time.values, sig]).T, n_out=n_out).T[1]
        ds_signals.append(down_sampled)

    # Unload the data
    example.unload_data()

    return pd.DataFrame(np.concatenate(ds_signals, axis=0)).T


