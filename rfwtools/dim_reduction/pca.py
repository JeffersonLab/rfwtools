import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def do_pca_reduction(feature_df, metadata_cols, n_components=3, standardize=True, **kwargs):
    """Performs PCA on subset of columns of feature_df and maintains some example info in results.

    Args:
        feature_df (DataFrame) - DataFrame containing example information and feature data.
        metadata_cols (list(str)) - The column names of feature_df that contain the metadata of the events (labels,
                                    etc.).  All columns not listed in event_cols are used in PCA analysis.
        n_components (int) - The number of primary components to return
        standardize (bool) - Should the features be standardized (i.e. (x-mean)/stdev)?
        kwargs (dict) - A dictionary of keyword parameter name/values to be passed to sklearn.decomposition.PCA

    Returns: (DataFrame) - A DataFrame containing the PCA output (pc1, pc2, ..., pcN) and specified event_cols.  One
                           row per event.
    """

    # Pull out the metadata portion
    y = feature_df[metadata_cols].copy().reset_index(drop=True)

    # Figure out if we have enough data to even try it.  If not return a DataFrame with pc1,... columns as NaNs
    columns = ["pc" + str(i) for i in range(1, n_components + 1)]
    if len(feature_df) < n_components:
        return pd.concat([y, pd.DataFrame(data=None, columns=columns)], axis=1)

    # Get the feature columns only
    x = feature_df.drop(metadata_cols, axis=1).copy().reset_index(drop=True)

    if standardize:
        scaler = StandardScaler(copy=False)
        x = scaler.fit_transform(x)

    # Do PCA reduction
    pca = PCA(n_components=n_components, **kwargs)
    principal_components = pca.fit_transform(x)

    # Get the results in a more "standard" format
    principal_df = pd.DataFrame(data=principal_components, columns=columns).reset_index(drop=True)
    pca_df = pd.concat([y, principal_df], axis=1)

    # Return both the results, and the fitted PCA object
    return pca_df, pca
