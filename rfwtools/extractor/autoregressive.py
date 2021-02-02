import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from sklearn import preprocessing
from rfwtools.utils import get_signal_names
from .utils import get_example_data


def autoregressive_extractor(ex, normalize=True, max_lag=5, signals=None, query=None):
    """Uses statsmodels to generate autoregressive model of each waveform.  AR coefficients are returned as features.

    This function handles loading and unloading the Example's data.

    Note: these features have historically been used for both cavity and fault type model training.

    Args:
        ex (Example) - The example for which we are generating features
        normalize (bool) - Should each waveform be normalized prior to autoregressive model fitting
        max_lag (int) - The number of AR parameters to fit (plus one for a bias/constant term)
        signals (list) - The list of signals to model (e.g. ["1_GMES", ...].  If None a default (1-8, GMES, CASK, CRFP,
                         DETA2) set is used.
        query (str) - Argument passed to the ex.event_df to filter data prior to feature extraction, e.g. "Time <= 0".
    """

    # Get the data from the Example
    event_df = get_example_data(ex, query)

    # List of signals for feature extraction
    sel_col = signals
    if signals is None:
        sel_col = get_signal_names(cavities=('1', '2', '3', '4', '5', '6', '7', '8'),
                                   waveforms=("GMES", "GASK", "CRFP", "DETA2"))

    # We only need to create this once.  Every time we "fit_transform" we update the values.
    signal_scaler = None
    if normalize:
        signal_scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

    # Feature extraction
    feature_names = []
    coefficients = None
    for colName in sel_col:
        signal = event_df[colName].values

        # Process the signal
        parameters = __process_signal(signal, max_lag=max_lag, scaler=signal_scaler)

        # Accumulate parameters from each signal to obtain the feature vector.
        if coefficients is None:
            coefficients = parameters
        else:
            coefficients = np.append(coefficients, parameters, axis=0)

        # Collect feature names
        feature_names = feature_names + [colName + '_AR_' + str(i) for i in range(0, max_lag + 1)]

    # transform to make the df a column vector
    feature_df = pd.DataFrame(coefficients).T

    # Add feature names as df column names
    feature_df.columns = feature_names

    return feature_df


def autoregressive_extractor_faulted_cavity(ex, normalize=True, max_lag=5, waveforms=None, query=None):
    """Generates AR features for waveforms of the cavity labeled as faulted.

    This function handles loading and unloading the Example's data.  No data after the fault is considered.  Returns
    None if cavity_label is '0' (Multi Cav tur off) since only a single cavity is to be considered.

    Note: these features have historically been used for visualizations as they are faster to computer and have a lower
          dimensionality.

    Args:
        ex (Example) - The example for which we are generating features
        normalize (bool) - Should each waveform be normalized prior to autoregressive model fitting
        max_lag (int) - The number of AR parameters to fit (plus one for a bias/constant term)
        waveforms (list) - The list of waveforms to model (e.g. ["GMES", ...].  If None a default (GMES, CASK, CRFP,
                         DETA2) set is used.  Note: the faulted cavity (cavity_label) is the only cavity used.
        query (str) - Argument passed to the ex.event_df to filter data prior to feature extraction, e.g. "Time <= 0".

    Returns: (DataFrame
    """

    if ex.cavity_label == "0":
        return None

    # Get the data from the Example
    event_df = get_example_data(ex, query)

    # List of signals for feature extraction
    sel_col = get_signal_names(cavities=ex.cavity_label, waveforms=("GMES", "GASK", "CRFP", "DETA2"))
    if waveforms is not None:
        sel_col = get_signal_names(cavities=ex.cavity_label, waveforms=waveforms)

    # We only need to create this once.  Every time we "fit_transform" we update the values.
    signal_scaler = None
    if normalize:
        signal_scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

    # Feature extraction
    feature_names = []
    coefficients = None
    for colName in sel_col:
        signal = event_df[colName].values

        # Process the signal to get the AR coefficients
        parameters = __process_signal(signal, max_lag=max_lag, scaler=signal_scaler)

        # Accumulate parameters from each signal to obtain the feature vector.
        if coefficients is None:
            coefficients = parameters
        else:
            coefficients = np.append(coefficients, parameters, axis=0)

        # Collect feature names - Make sure to strip of the cavity label since this has only the cavity that faulted
        feature_names = feature_names + [colName[2:] + '_AR_' + str(i) for i in range(0, max_lag + 1)]

    # transform to make the df a column vector
    feature_df = pd.DataFrame(coefficients).T

    # Add feature names as df column names
    feature_df.columns = feature_names

    return feature_df


def __process_signal(signal, max_lag, scaler=None):
    """Internal function for calculating AR features of a single signal

    Args:
        signal (np.array) - The values of the signal to be fitted by AR coefficients
        max_lag (int) - The number of AR parameters to fit (plus one for a bias/constant term)
        scaler (sklearn.StandardScaler) - Scaler used to standardized the signal.  If None, no scaling is performed.

    """

    # if the signal is constant values, features are zeros
    if np.size(np.unique(signal)) == 1:
        parameters = np.zeros(max_lag + 1, dtype=np.float64)
    else:

        if scaler is not None:
            signal = np.squeeze(scaler.fit_transform(signal.reshape(-1, 1)))

        # AR model fitting - using old_names=True to suppress warning about future deprecation of kwargs for AutoReg
        # after v0.12
        model = AutoReg(signal, lags=max_lag, trend='ct', old_names=True)
        model_fit = model.fit()

        # If AR model fits the signal with less than maxLag + 1 parameters, pad the rest with zeros
        # If AR model uses more than maxLag + 1, choose the first maxLag + 1 parameters as features
        if np.shape(model_fit.params)[0] < max_lag + 1:
            parameters = np.pad(model_fit.params, (0, max_lag + 1 - np.shape(model_fit.params)[0]),
                                'constant', constant_values=0)
        elif np.shape(model_fit.params)[0] > max_lag + 1:
            parameters = model_fit.params[: max_lag + 1]
        else:
            parameters = model_fit.params

    return parameters
