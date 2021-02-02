import re

from .utils import get_example_data

from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute

# TODO - make a unit test for this
from ..utils import get_signal_names


def tsfresh_extractor(example, query=None, impute_function=impute, disable_progress_bar=True, n_jobs=0,
                      default_fc_parameters=EfficientFCParameters(), **kwargs):
    """Uses tsfresh to extract features.

    All parameters not listed below shadow tsfresh.extract_features parameters and are passed to that function.

    Args:
        example (Example) - The Example for which features are extracted
        query (str) - Argument passed to the ex.event_df to filter data prior to feature extraction, e.g. "Time <= 0".
        **kwargs (dict) - All other key word arguments are passed directly to tsfresh.extract_features

    """

    # Get the Example's data
    event_df = get_example_data(example, query)

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


def tsfresh_extractor_faulted_cavity(example, waveforms=None, query=None, impute_function=impute,
                                     disable_progress_bar=True, n_jobs=0, default_fc_parameters=EfficientFCParameters(),
                                     **kwargs):
    """Uses tsfresh to extract features for only the cavity that faulted.  Returns None if cavity_label=='0'.

    All parameters not listed below shadow tsfresh.extract_features parameters and are passed to that function.

    Args:
        example (Example) - The Example for which features are extracted
        waveforms (list(str)) - A list of waveform names to extract features from.
                                Default: ['GMES', 'GASK', 'CRFP', 'DETA2']
        query (str) - Argument passed to the ex.event_df to filter data prior to feature extraction, e.g. "Time <= 0".
        **kwargs (dict) - All other key word arguments are passed directly to tsfresh.extract_features

    """

    if example.cavity_label == "0":
        return None

    # Get the Example's data
    event_df = get_example_data(example, query)

    # List of signals for feature extraction
    sel_col = get_signal_names(cavities=example.cavity_label, waveforms=("GMES", "GASK", "CRFP", "DETA2"))
    if waveforms is not None:
        sel_col = get_signal_names(cavities=example.cavity_label, waveforms=waveforms)

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
