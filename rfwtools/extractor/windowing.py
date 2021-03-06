"""This module provides the ability to grab time-based windows of signals, and provide labels to them.

Typically, these will be used by DataSet.produce_feature_set().  However there is no reason why these can't be run
externally.

Basic Usage Example:
::

    from rfwtools.data_set import DataSet
    from rfwtools.extractor.windowing import window_extractor, window_extractor_metadata

    # Setup a DataSet object and get some example data to work with
    ds = DataSet()
    ds.load_example_set_csv("my_example_set.csv")

    # Get a single example to work on
    ex = ds.example_set.loc[0, 'example']

    # Run on one example with defaults (start at Time == -500, and include the next 100 samples)
    window_extractor(ex, windows={'my_window': -500}, n_samples=100)

    # Run on one example with only 2 signals being processed
    window_extractor(ex, windows={'my_window': -500}, n_samples=100,
                     signals=['1_GMES', '1_PMES'])

    # Produce one window for stable running and one window for impending fault
    window_extractor(ex, windows={'stable': -1536, 'impending': -105}, n_samples=500)

    # Run this on every example in the example set and produce a corresponding feature set for pre-fault signal data.
    ds.produce_feature_set(window_extractor, windows={'my_window': -500}, n_samples=100)

    # Update the metdata_columns to reflect the additional columns generated by the window_extractor
    ds.feature_set.update_metadata_columns(window_extractor_metadata)

"""
import re
from typing import List, Dict, Tuple, Union

import pandas as pd
from scipy import signal as sgl
from sklearn.preprocessing import StandardScaler
import numpy as np

from .utils import get_example_data
from ..utils import get_signal_names
from ..example import Example

#: A list of metadata column names generated by the window_extractor
window_extractor_metadata = ['window_label', 'window_start', 'n_samples', 'window_min', 'window_max',
                             'window_standardized', 'window_downsampled', 'window_downsample_size']


def window_extractor(example: Example, windows: Dict[str, float], n_samples: int, signals: List[str] = None,
                     standardize: bool = True, downsample: bool = False, ds_kwargs: dict = {'num': 256, 'axis': 0},
                     query: str = None) -> pd.DataFrame:
    """
    Extract labeled time windows of a Example's event_df.  Will produce one feature row for each key in windows.

    Windows are  left inclusive, right exclusive.

    One use of this is to extract two windows of data.  One relating to 'stable' running, another to fault 'impending'.

    Please note the order of operations.  First query is applied to the Example's event_df.  Then each signal is
    standardized.  Then the standardized signal is down sampled using scipy.signal.resample.

    Arguments:
        example:
            The example on which to operate
        windows:
            A dictionary keyed on window labels (e.g., 'stable', 'impending fault', etc.) with values that are the start
            of the window.
        n_samples:
            The number of samples to include from start.
        signals:
            An explicit list of the example's columns to be down sampled (e.g., "1_GMES").  If None, then
            [GMES, CRFP, DETA2, GASK] will be used for all 8 cavities.
        standardize:
            If True, each signal is z-score standardized ( (x-u)/s) ).  No change if False.  Each window is standardized
            independently.
        downsample:
            Should the signals be down sampled.
        ds_kwargs:
            Keyword arguments that will be passed to the scipy.signal.resample routine, if downsample is True
        query:
            Argument passed to the ex.event_df to filter data prior to feature extraction, e.g. "Time <= 0".

    Returns:
         A DataFrame with a row each requested window.  Has a window_label column and a 'step_#' column for each Time
         sample, where # is the number of samples from the start of the window (after down sampling, if requested).
    """
    # List of signals for feature extraction
    if signals is None:
        signals = get_signal_names(cavities=['1', '2', '3', '4', '5', '6', '7', '8'],
                                   waveforms=["GMES", "GASK", "CRFP", "DETA2"])

    # Call the real workhorse
    return _window_extractor(example=example, windows=windows, n_samples=n_samples, signals=signals,
                             standardize=standardize, downsample=downsample, ds_kwargs=ds_kwargs, query=query)


def window_extractor_faulted_cavity(example: Example, windows: Dict[str, float], n_samples: int, waveforms: List[str],
                                    standardize: bool = True, downsample: bool = False,
                                    ds_kwargs: dict = {'num': 256, 'axis': 0},
                                    query: str = None) -> Union[pd.DataFrame, None]:
    """
    Extract labeled time windows of a Example's event_df for the labeled cavity.  None if cavity_label is '0'

    Will produce one feature row for each key in windows.  Windows are  left inclusive, right exclusive.

    One use of this is to extract two windows of data.  One relating to 'stable' running, another to fault 'impending'.

    Please note the order of operations.  First query is applied to the Example's event_df.  Then each signal is
    standardized.  Then the standardized signal is down sampled using scipy.signal.resample.

    Arguments:
        example:
            The example on which to operate
        windows:
            A dictionary keyed on window labels (e.g., 'stable', 'impending fault', etc.) with values that are the start
            of the window.
        n_samples:
            The number of samples to include from start.
        waveforms:
            An explicit list of the waveforms to be down sampled (e.g., "GMES").  If None, then
            [GMES, CRFP, DETA2, GASK] will be used for the cavity labeled as faulting.
        standardize:
            If True, each signal is z-score standardized ( (x-u)/s) ).  No change if False.  Each window is standardized
            independently.
        downsample:
            Should the signals be down sampled.
        ds_kwargs:
            Keyword arguments that will be passed to the scipy.signal.resample routine, if downsample is True
        query:
            Argument passed to the ex.event_df to filter data prior to feature extraction, e.g. "Time <= 0".

    Returns:
        A DataFrame with a row each requested window.  Has a window_label column and a 'Sample_#_<waveform>' column for
        each Time sample, where # is the number of samples from the start of the window (after down sampling, if
        requested).  None if cavity_label is '0'.
    """

    # Filter out multi cav turn offs
    if example.cavity_label == '0':
        return None

    # Define the signals to call
    if waveforms is None:
        waveforms = ["GMES", "CRFP", "DETA2", "GASK"]
    signals = get_signal_names(example.cavity_label, waveforms=waveforms)

    # Call the real workhorse
    out = _window_extractor(example=example, windows=windows, n_samples=n_samples, signals=signals,
                            standardize=standardize, downsample=downsample, ds_kwargs=ds_kwargs, query=query)

    # Strip out the cavity number from Sample_... column names and return
    return out.rename(columns=_col_rename_remove_cavity)


def _col_rename_remove_cavity(col_name):
    """Rename Sample_ column to remove the cavity number.  Leave other columns unchanged."""
    p = re.compile(r"^(Sample_\d+)(_\d)(_\w+)")
    m = p.match(col_name)
    if m:
        return f"{m.group(1)}{m.group(3)}"
    else:
        return col_name


def _window_extractor(example: Example, windows: Dict[str, float], n_samples: int, signals: List[str],
                      standardize: bool = True, downsample: bool = False, ds_kwargs: dict = {'num': 256, 'axis': 0},
                      query: str = None) -> pd.DataFrame:
    """
    Extract labeled time windows of a Example's event_df.  Will produce one feature row for each key in windows.

    Windows are  left inclusive, right exclusive.

    One use of this is to extract two windows of data.  One relating to 'stable' running, another to fault 'impending'.

    Please note the order of operations.  First query is applied to the Example's event_df.  Then each signal is
    standardized.  Then the standardized signal is down sampled using scipy.signal.resample.

    Arguments:
        example:
            The example on which to operate
        windows:
            A dictionary keyed on window labels (e.g., 'stable', 'impending fault', etc.) with values that are the start
            of the window.
        n_samples:
            The number of samples to include from start.
        signals:
            An explicit list of the example's columns to be down sampled (e.g., "1_GMES").  If None, then
            [GMES, CRFP, DETA2, GASK] will be used for all 8 cavities.
        standardize:
            If True, each signal is z-score standardized ( (x-u)/s) ).  No change if False.  Each window is standardized
            independently.
        downsample:
            Should the signals be down sampled.
        ds_kwargs:
            Keyword arguments that will be passed to the scipy.signal.resample routine, if downsample is True
        query:
            Argument passed to the ex.event_df to filter data prior to feature extraction, e.g. "Time <= 0".

    Returns:
         A DataFrame with a row each requested window.  Has a window_label column and a 'step_#' column for each Time
         sample, where # is the number of samples from the start of the window (after down sampling, if requested).
    """

    # Get the example query
    df = get_example_data(example, query)

    out_list = []
    for label in windows.keys():
        # Verify that we can actually get the requested window
        start = windows[label]
        start_i = int(df.query(f"Time >= {start}").index.min())
        end_i = int(start_i + n_samples)

        # Check if the requested window is available
        i_min = df.index.min()
        i_max = df.index.max()
        t_min = df.Time[i_min]
        t_max = df.Time[i_max]
        if i_min > start_i or i_max < end_i:
            raise RuntimeError(f"Example's time range ([{t_min}, {t_max}] does not contain window ({start} + "
                               f"{n_samples} steps).")

        # Get the window's data, and adjust the Time column to be relative to the start of the window.  Explicitly get
        # a copy of df to avoid a SettingWithCopy warning.  We don't want to change the original, so this is right.
        window_df = df.copy()
        window_df = window_df.iloc[start_i:end_i, :]
        window_t_min = window_df.Time.min()
        window_t_max = window_df.Time.max()
        window_df.Time = window_df.Time - start

        # Start a single row DataFrame for this window
        if downsample:
            window_downsample_size = ds_kwargs['num']
        else:
            window_downsample_size = None

        out_df = pd.DataFrame({'window_label': pd.Categorical([label], categories=windows.keys()),
                               'window_start': start, 'n_samples': n_samples, 'window_min': window_t_min,
                               'window_max': window_t_max, 'window_standardized': standardize,
                               'window_downsampled': downsample, 'window_downsample_size': window_downsample_size})

        # Setup on common scaler device.  It will be continually overwritten
        if standardize:
            scaler = StandardScaler(copy=True, with_mean=True, with_std=True)

        # Process each signal one at a time
        for signal in signals:
            signal_values = np.expand_dims(window_df[signal].values, axis=1)

            # Standardize if requested
            if standardize:
                signal_values = scaler.fit_transform(signal_values)

            # Down sample if requested
            if downsample:
                signal_values = sgl.resample(signal_values, **ds_kwargs)

            # Append the signal values to the row.  Each value to a column.  Column names should look like
            # "Sample_#_<cav>_<waveform>", with values that are a list containing the values at each sample step
            tmp_df = pd.DataFrame(data=signal_values.reshape((1, len(signal_values))),
                                  columns=[f"Sample_{x}_{signal}" for x in range(1, len(signal_values) + 1)])
            out_df = pd.concat([out_df, tmp_df], axis=1)

        # Add the window's one rowed DataFrame to the list of DataFrame to output
        out_list.append(out_df)

    # Concatenate all of the one row DataFrames into a single multi-row DataFrame
    return pd.concat(out_list, ignore_index=True)
