"""The module encapsulates the functionality around computed features of an example set.

Typically a FeatureSet is created by DataSet.produce_feature_set().  Once created the class provides some tools for
easy dimensionality reduction and visualization.

Basic Usage Examples:

Creation through a DataSet workflow:
::

    from rfwtools.data_set import DataSet
    from rfwtools.extractor.autoregressive import autoregressive_extractor

    ds = DataSet(label_files=['my-sample-labels.txt'])
    ds.produce_example_set()
    ds.produce_feature_set(autoregressive_extractor)

    fs = ds.feature_set
    fs.do_pca_reduction(n_components=10)
    fs.display_2d_scatterplot(query="zone=='1L25'", title="1L25 PCA subset")
    fs.save_csv("my_feature_set.csv")

Loading from file:
::

    from rfwtools.feature_set import FeatureSet

    fs = FeatureSet()
    fs.load_csv("my_feature_set.csv")
    fs.do_pca_reduction()
    fs.display_2d_scatterplot(query="zone=='1L25'", title="1L25 PCA subset")

"""


import os
from typing import List, Tuple

import pandas as pd

from rfwtools.dim_reduction import pca
from rfwtools.visualize import scatterplot
from rfwtools.config import Config
from rfwtools.example_set import ExampleSet


class FeatureSet(ExampleSet):
    """A class for managing common operations on a collection of labeled faults and associated features

    This class is an extension on ExampleSet meant to handle additional analysis operations.
    """

    def __init__(self, df: pd.DataFrame = None, filename: str = None, in_dir: str = None, sep: str = ",",
                 name: str = "", metadata_columns: List[str] = None):
        """Construct a FeatureSet.  Can use either a DataFrame or CSV-like file.

        Arguments:
            df:
                A DataFrame containing the FeatureSet data to include.  The following columns must be included -
                ['zone', 'dtime', 'cavity_label', 'fault_label', 'label_source'].  Any additional columns will be
                treated as the features.  Note: A copy of df is saved.
            filename:
                The filename to load.  Will be relative to in_dir
            in_dir:
                The directory to find the CSV-like file in.  Defaults to Config().output_dir
            sep:
                Delimiter string used by Pandas to parse given CSV-like file
            name:
                A string that may be used to help identify this FeatureSet
            metadata_columns:
                A list of the names of the columns that are metadata (e.g., "zone" or "cavity_label").  Any supplied
                column names are in addition to the mandatory columns for an ExampleSet.  metadata_columns are treated
                as required columns.
        """

        # Setup the Example
        super().__init__()

        #: These columns are required in internal DataFrames and are excluded from analysis routines.
        self.metadata_columns = None
        self.update_metadata_columns(metadata_columns=metadata_columns)

        if df is not None:
            # Make sure the DataFrame has the metadata columns.  The
            if 'example' not in df.columns.to_list():
                if not self.has_required_columns(df, dtypes=True, skip_example=True):
                    raise ValueError("A column is missing from the supplied DataFrame or has wrong dtype")
                df['example'] = df.apply(self._Example_from_row, axis=1, raw=False)
            else:
                for col in self.metadata_columns:
                    if col not in df.columns:
                        raise ValueError("A column is missing from the supplied DataFrame or has wrong dtype")

            # Make a standard column order.  All metadata columns first, followed by feature columns
            df = df[self.metadata_columns + df.drop(columns=self.metadata_columns).columns.to_list()]

            # Import the DataFrame into the ExampleSet's internal DataFrame
            self.update_example_set(df)

        elif filename is not None:
            self.load_csv(filename=filename, in_dir=in_dir, sep=sep)

        # A brief human friendly name for this FeatureSet
        self.name = name

        #: The pca reduced feature data, one example per row
        self.__pca_df = None

        #: The pca model.  Either None or is the fitted sklearn PCA object.  This is left publicly accessible so users
        #: have access for custom analysis or visualization (e.g., transforming future examples).  Users beware
        #: modifying this!
        self.pca = None

    def load_csv(self, filename: str, in_dir: str = None, sep: str = ',', metadata_columns: List[str] = None) -> None:
        """Read in a CSV file that has FeatureSet data.  Relative to in_dir if filename is str.

        Arguments:
            filename:
                The filename to load.  Will be relative in_dir
            in_dir:
                The directory to find the file in.  Defaults to Config().output_dir
            sep:
                Delimiter string used by Pandas to parse given "csv" file
            metadata_columns:
                A list of column names to treat as metadata.  This updates the FeatureSet's list.  No changes are made
                if it is None.
        """
        
        # Update the column info if indicated
        if metadata_columns is not None:
            self.update_metadata_columns(metadata_columns=metadata_columns)

        # Clear the dimensionality reduction attributes
        self.__pca_df = None
        self.pca = None

        # Load it into the parent's ExampleSet _example_df
        super().load_csv(filename=filename, in_dir=in_dir, sep=sep)

    def update_metadata_columns(self, metadata_columns: List[str]) -> None:
        """This updates the metadata columns and alters other related data structures.

        self.metadata_columns always include ExampleSet._mandatory_columns.  Does deduplication should you include those
        as well.

        Arguments:
            metadata_columns: A list of metadata columns.
        """
        if metadata_columns is None:
            # Use the required columns of the parent ExampleSet.  Since no extra required columns we can leave
            # self._req_columns alone
            self.metadata_columns = self.get_required_columns()
        else:
            # Here we're assuming that all of the metadata columns should be required
            # This would have the old metadata columns in it, and it is returned by get_required_columns().
            self._req_columns = []

            # Make sure we don't have duplicate column names
            m_uniq = []
            for col in (self.get_required_columns() + metadata_columns):
                if col not in m_uniq:
                    m_uniq.append(col)
            self.metadata_columns = m_uniq

            self._req_columns = metadata_columns

    def get_pca_df(self) -> pd.DataFrame:
        """Get a copy of the PCA reduction as a DataFrame.  Will be None if the reduction has not been done.

        Each example is on it's own row with it's primary components.

        Returns:
            A copy of the PCA results along with the example's metadata.
        """
        if self.__pca_df is None:
            return None
        else:
            return self.__pca_df.copy()

    def update_example_set(self, df: pd.DataFrame, metadata_columns: List[str] = None) -> None:
        """Update the _example_df and blanks other internal data derived from it.

        Arguments:
            df:
                A DataFrame containing an example per row with additional feature information.  Must be valid for this
                FeatureSet.
            metadata_columns:
                A new list of metadata columns for df
        """
        if metadata_columns is not None:
            self.update_metadata_columns(metadata_columns=metadata_columns)

        super().update_example_set(df)
        self.__pca_df = None
        self.pca = None

    def do_pca_reduction(self, metadata_cols: List[str] = None, report: bool = True, n_components: float = 3,
                         **kwargs) -> None:
        """Perform PCA on subset of columns of example_df and maintain example metadata in results.

        The results are accessible through the get_pca_df() method and the fitted model through the pca attribute.

        Arguments:
            metadata_cols:
                The column names of feature_df that contain the metadata of the events (labels, etc.).  All columns not
                listed in event_cols are used in PCA analysis.  If None, it defaults to the values supplied at
                construction of FeatureSet.
            report:
                Should a report of explained variance be printed
            n_components:
                The number of principal components to calculate.
            **kwargs:
                Remaining keyword arguments will be passed to sklearn.decomposition.PCA
        """
        # Setup data for doing PCA.  x is the feature data to be reduced
        if metadata_cols is None:
            # Pandas lets you use a tuple as a key.  Need as a list.
            metadata_cols = list(self.metadata_columns)

        # Do the PCA dimensionality reduction
        self.__pca_df, self.pca = pca.do_pca_reduction(self._example_df, metadata_cols=metadata_cols,
                                                       n_components=n_components, **kwargs)

        # Print explained variance ratio if requested
        if report:
            print("Explained Variance Ratio")
            print(self.pca.explained_variance_ratio_)

    def display_2d_scatterplot(self, technique: str = "pca", alpha: float = 0.8, s: int = 25, title: str = None,
                               figsize: Tuple[int, int] = (12, 12), query: str = None, **kwargs) -> None:
        """Display a two-dimensional scatterplot of the dimensionally reduced feature set.

        If the type specified has not already been generated, an exception is raised.  Note: consider passing
        hue=<pca_df column_name> and/or style=<pca_df column_name> to control which column colors/styles each point.

        Arguments:
            technique: The type of dim reduction data to display.  Currently the only supported option is pca.
            alpha: Controls point transparency
            s: Controls point size
            title: The title of the scatterplot
            figsize: The two dimensions of the size of the figure.  Passed to plt.figure.
            query: A pd.DataFrame.query() expr argument.  Used to subset the data prior to plotting.
            **kwargs: All remaining parameters are passed directly to the scatterplot command
        """

        # Figure out which type of plot we're showing
        df, x, y = self.__select_technique_data(technique=technique)

        # Subset the data if requested
        if query is not None:
            df = df.query(query)

        if title is None:
            title = f"{self.name} ({technique}, n={len(df)})"

        # Plot the figure
        scatterplot.scatterplot(data=df, x=x, y=y, title=title, alpha=alpha, s=s, figsize=figsize, **kwargs)

    def __select_technique_data(self, technique):
        """Convenience method for getting the x,y plotting columns and total dataset for a given dim reduction."""
        if technique == "pca":
            if self.__pca_df is None:
                raise RuntimeError("Internal pca_df is None.  Must first run do_pca_reduction().")

            # Filter the DataFrame if requested
            df = self.__pca_df
            x = 'pc1'
            y = 'pc2'
        else:
            raise ValueError("Only technique='pca' is supported")

        return df, x, y

    def __eq__(self, other):
        """Check equality between FeatureSets based on feature_df and metadata_columns."""

        # Short circuit checks
        if type(other) is not FeatureSet:
            return False
        if self is other:
            return True

        # Check the metadata columns
        if self.metadata_columns != other.metadata_columns:
            return False

        # Check the feature DataFrame
        if not self._example_df.equals(other._example_df):
            return False

        return True
