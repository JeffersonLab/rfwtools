"""This module provides tsfresh-based time series statistical feature extraction tools.

Typically, these will be used by DataSet.produce_feature_set().  However there is no reason why these can't be run
externally.

Basic Usage Example:
::

    from rfwtools.data_set import DataSet
    from rfwtools.extractor.tsf import tsfresh_extractor

    # Setup a DataSet object and get some example data to work with
    ds = DataSet()
    ds.load_example_set_csv("my_example_set.csv")

    # Get a single example to work on
    ex = ds.example_set.loc[0, 'example']

    # Run on one example with defaults
    tsfresh_extractor(ex)
    # Run on one example with only 2 signals being processed
    tsfresh_extractor(ex, signals=['1_GMES', '1_PMES'])
    # Run on one example, but only include values before the fault on set.
    tsfresh_extractor(ex, query="Time < 0")

    # Run this on every example in the example set and produce a corresponding feature set for pre-fault signal data.
    ds.produce_feature_set(tsfresh_extractor, query="Time < 0")

"""

import re
import pandas as pd
from typing import Union, List
from .utils import get_example_data
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute
from ..example import Example
from ..utils import get_signal_names


def tsfresh_extractor(example: Example, signals: List[str] = None, query: str = None,
                      impute_function: Union[callable, None] = impute, disable_progress_bar: bool = True,
                      n_jobs: int = 0, default_fc_parameters: dict = None, **kwargs) -> pd.DataFrame:
    """Use tsfresh to extract features specified.

    This is a thin wrapper over tsfresh.feature_extraction.extraction.extract_features.  See that method for more
    details.

    Arguments:
        example:
            The Example for which features are extracted
        signals:
            A list of signal names to extract features from. Default: combination of cavities 1-8 and waveforms =
            ['GMES', 'GASK', 'CRFP', 'DETA2']
        query:
            Argument passed to the ex.event_df to filter data prior to feature extraction, e.g. "Time <= 0".
        impute_function:
            The function used to impute missing values about the data
        disable_progress_bar:
            Should the progress bar be displayed?
        n_jobs:
            The number of jobs should be run concurrently.  Defaults to zero, which disables parallelization.
        default_fc_parameters:
            mapping of feature calculator names to parameters.  If None, defaults to EfficientFCParameters().  See
            tsfresh.feature_extraction.extraction.extract_features for more details.
        **kwargs:
            All other key word arguments are passed directly to tsfresh.extract_features

    Returns:
        A DataFrame of the calculated features.
    """

    # Get the Example's data
    event_df = get_example_data(example, query)

    # List of signals for feature extraction
    sel_col = signals
    if signals is None:
        sel_col = get_signal_names(cavities=['1', '2', '3', '4', '5', '6', '7', '8'],
                                   waveforms=["GMES", "GASK", "CRFP", "DETA2"])

    # Set the default feature parameters
    if default_fc_parameters is None:
        default_fc_parameters = EfficientFCParameters()

    # Get the data that matches the request
    event_df = event_df[["Time"] + sel_col]

    # Add the ID column tsfresh wants.  Mostly useless here since we only give tsfresh a single example at a time.
    event_df.insert(loc=0, column='id', value=1)

    # Do the feature extraction
    feature_df = extract_features(event_df.astype('float64'),
                                  column_id="id",
                                  column_sort="Time",
                                  impute_function=impute_function,
                                  default_fc_parameters=default_fc_parameters,
                                  disable_progressbar=disable_progress_bar,
                                  n_jobs=n_jobs,
                                  **kwargs
                                  ).reset_index()
    feature_df.drop(columns='index', inplace=True)
    return feature_df


def tsfresh_extractor_faulted_cavity(example: Example, waveforms: List[str] = None, query: str = None,
                                     impute_function: Union[callable, None] = impute, disable_progress_bar: bool = True,
                                     n_jobs: int = 0, default_fc_parameters: dict = None,
                                     **kwargs) -> Union[pd.DataFrame, None]:
    """Use tsfresh to extract features for only the cavity that faulted.  Returns None if cavity_label=='0'.

    This is a thin wrapper over tsfresh.feature_extraction.extraction.extract_features.  See that method for more
    details.

    Arguments:
        example:
            The Example for which features are extracted
        waveforms:
            A list of waveform names to extract features from. Default is ['GMES', 'GASK', 'CRFP', 'DETA2'].
        query:
            Argument passed to the ex.event_df to filter data prior to feature extraction, e.g. "Time <= 0".
        impute_function:
            The function used to impute missing values about the data
        disable_progress_bar:
            Should the progress bar be displayed?
        n_jobs:
            The number of jobs should be run concurrently.  Defaults to zero, which disables parallelization.
        default_fc_parameters:
            mapping of feature calculator names to parameters.  If None, defaults to EfficientFCParameters().  See
            tsfresh.feature_extraction.extraction.extract_features for more details.
        **kwargs:
            All other key word arguments are passed directly to tsfresh.extract_features

    Returns:
        A DataFrame of the calculated features or None if cavity_label=='0'.
    """

    if example.cavity_label == "0":
        return None

    # Get the Example's data
    event_df = get_example_data(example, query)

    # List of signals for feature extraction
    sel_col = get_signal_names(cavities=example.cavity_label, waveforms=["GMES", "GASK", "CRFP", "DETA2"])
    if waveforms is not None:
        sel_col = get_signal_names(cavities=example.cavity_label, waveforms=waveforms)

    # Set the default feature parameters
    if default_fc_parameters is None:
        default_fc_parameters = EfficientFCParameters()

    # Get the requested columns for the cavity that faulted.  Then drop the cavity id from the column name so features
    # for all examples will have same column names.
    event_df = event_df[["Time"] + sel_col]
    event_df = event_df.rename(lambda x: re.sub('\d_', '', x), axis='columns')

    # Add the ID column tsfresh wants.  Mostly useless here since we only give tsfresh a single example at a time.
    event_df.insert(loc=0, column='id', value=1)

    # Do the feature extraction
    feature_df = extract_features(event_df.astype('float64'),
                                  column_id="id",
                                  column_sort="Time",
                                  impute_function=impute_function,
                                  default_fc_parameters=default_fc_parameters,
                                  disable_progressbar=disable_progress_bar,
                                  n_jobs=n_jobs,
                                  **kwargs
                                  ).reset_index()
    feature_df.drop(columns='index', inplace=True)
    return feature_df
