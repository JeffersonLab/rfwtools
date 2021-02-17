import os
import pandas as pd

from rfwtools.dim_reduction import pca
from rfwtools.visualize import scatterplot
from rfwtools.config import Config
from rfwtools.example_set import ExampleSet


class FeatureSet(ExampleSet):
    """A class for managing common operations on a collection of labeled faults and associated features

    This class is a light wrapper on a Pandas DataFrame.
    """

    def __init__(self, df=None, filename=None, in_dir=None, sep=None, name="", metadata_columns=None):
        """Construct a FeatureSet.  Use the supplied DataFrame, or load filename if no df argument is given

        Args:
            df (DataFrame): A DataFrame containing the FeatureSet data to include.  The following columns must be
                            included - ['zone', 'dtime', 'cavity_label', 'fault_label', 'label_source'].  Any
                            additional columns will be treated as the features.  Note: A copy of df is saved.
            filename (str) - The filename to save.  Will be relative in_dir
            in_dir (str) - The directory to find the file in.  Defaults to Config().output_dir
            sep (str) - Delimiter string used by Pandas to parse given "csv" file
            name (str): A string that may be used to help identify this FeatureSet in plot titles, etc.
            metadata_columns (list(str)): A list of the names of the columns that are metadata (e.g., "zone" or
                                          "cavity_label").  If None, a standard set is assumed.
        """

        if metadata_columns is None:
            self.metadata_columns = ['zone', 'dtime', 'cavity_label', 'fault_label', 'label_source']
        else:
            self.metadata_columns = metadata_columns

        if df is not None:
            for col in self.metadata_columns:
                if col not in df.columns:
                    raise ValueError(f"column {col} is missing from the supplied DataFrame")

            # The feature data, one example per row
            self.__feature_df = df.copy()

        elif filename is not None:
            self.load_csv(filename=filename, in_dir=in_dir, sep=sep)

        # A brief human friendly name for this FeatureSet
        self.name = name

        # The pca reduced feature data, one example per row
        self.__pca_df = None

        # The pca model.  Either None or is the fitted sklearn PCA object.  This is left publicly accessible so users
        # have access for custom analysis or visualization (e.g., transforming future examples).  Users beware
        # modifying this!
        self.pca = None

    def save_csv(self, filename, out_dir=None, sep=','):
        """Write out the FeatureSet data as a CSV file relative to out_dir.

        Args:
            filename (str) - The filename to save.  Will be relative out_dir
            out_dir (str) - The directory to save the file in.  Defaults to Config().output_dir
            sep (str) - Delimiter string used by Pandas to parse given "csv" file
        """
        if out_dir is None:
            out_dir = Config().output_dir
        self.__feature_df.to_csv(os.path.join(out_dir, filename), sep=sep, index=False)

    def load_csv(self, filename, in_dir=None, sep=',', metadata_columns=None):
        """Read in a CSV file that has FeatureSet data.

        Args:
            filename (str) - The filename to save.  Will be relative out_dir
            in_dir (str) - The directory to find the file in.  Defaults to Config().output_dir
            sep (str) - Delimiter string used by Pandas to parse given "csv" file
            metadata_columns (list) - A list of column names to treat as metadata.  This updates the FeatureSet's list.
        """
        if in_dir is None:
            in_dir = Config().output_dir
        df = pd.read_csv(os.path.join(in_dir, filename), sep=sep)

        if metadata_columns is not None:
            self.metadata_columns = metadata_columns

        # Test that we actually have a the columns we expect
        try:
            # If the columns don't exist, this will raise an exception
            df[self.metadata_columns]
        except KeyError as ex:
            print(df[['']])
            raise KeyError(f"File missing metadata columns -- {ex}")

        # Make it a standard FeatureSet
        FeatureSet.__standardize_df_format(df)

        self.__feature_df = df
        self.__pca_df = None

    def get_feature_df(self):
        """Return a copy of the FeatureSet as a DataFrame."""
        return self.__feature_df.copy()

    def get_pca_df(self):
        """Return a copy of the PCA reduction as a DataFrame.  Will be None if the reduction has not been done."""
        if self.__pca_df is None:
            return None
        else:
            return self.__pca_df.copy()

    def update_feature_df(self, df):
        """Updates the feature_df and blanks other internal data derived from it"""
        self.__feature_df = df.copy()
        self.__pca_df = None

    def do_pca_reduction(self, metadata_cols=None, report=True, n_components=3, **kwargs):
        """Performs PCA on subset of columns of feature_df and maintains some example info in results.

        Args:
            metadata_cols (list(str)) - The column names of feature_df that contain the metadata of the events (labels,
                                        etc.).  All columns not listed in event_cols are used in PCA analysis.  If None,
                                        it defaults to the values supplied at construction.
            report (bool) - Should a report of explained variance be printed
            n_components (int) - The number of principal components to calculate.
            **kwargs (dict) - Remaining keyword arguments will be passed to sklearn.decomposition.PCA

        Returns (PCA): Returns the PCA object after performing fit_transform()
        """
        # Setup data for doing PCA.  x is the feature data to be reduced
        if metadata_cols is None:
            # Pandas lets you use a tuple as a key.  Need as a list.
            metadata_cols = list(self.metadata_columns)

        # Do the PCA dimensionality reduction
        self.__pca_df, self.pca = pca.do_pca_reduction(self.__feature_df, metadata_cols=metadata_cols,
                                                       n_components=n_components, **kwargs)

        # Print explained variance ratio if requested
        if report:
            print("Explained Variance Ratio")
            print(self.pca.explained_variance_ratio_)

    def display_2d_scatterplot(self, technique="pca", alpha=0.8, s=25, title=None, figsize=(12, 12), query=None,
                               **kwargs):
        """Display a two-dimensional scatterplot of the dimensionally reduced feature set.

        If the type specified has not already been generated, an exception is raised.  Note: consider passing
        hue=<pd.Series> and/or style=<pd.Series> to control which column colors/styles each point.

        Args:
            technique (str): The type of dim reduction data to display.  Currently the only supported option is pca.
            alpha (float): Controls point transparency
            s (int): Controls point size
            title (str): The title of the scatterplot
            figsize (2-tuple(float)): The two dimensions of the size of the figure.  Passed to plt.figure.
            query (str): A pd.DataFrame.query() expr argument.  Used to subset the data prior to plotting.
            **kwargs (dict): All remaining parameters are passed directly to the scatterplot command
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
        if not self.__feature_df.equals(other.__feature_df):
            return False

        return True

    @staticmethod
    def __standardize_df_format(df, zones=None, cavity_labels=None, fault_labels=None):
        """Attempts to put a DataFrame in a 'standard' format.

        This affects IN-PLACE variables that should categoricals, datetime, float, etc. and creates the example column
        if not already present.  Columns are reordered.

        Args:
            df (pd.DataFrame) - The DataFrame to reformat

        Returns: None
        """

        # Seems like the datetime dtype doesn't want to stick
        df['dtime'] = df['dtime'].astype('datetime64[ns]')

        # Update the dtypes so that we get categories, etc. where it makes sense
        df['zone'] = df['zone'].astype('category')
        df['fault_label'] = df['fault_label'].astype('category')
        df['cavity_label'] = df['cavity_label'].astype('str')
        df['cavity_label'] = df['cavity_label'].astype('category')

        # Figure out what categories to enforce
        if zones is None:
            z = ExampleSet.known_zones
        else:
            z = zones
        if cavity_labels is None:
            cl = ExampleSet.known_cavity_labels
        else:
            cl = cavity_labels
        if fault_labels is None:
            fl = ExampleSet.known_fault_labels
        else:
            fl = fault_labels

        # Ensure a consistent set of category levels and their order.
        master = {
            'zone': z,
            'fault_label': fl,
            'cavity_label': cl
        }

        # Add any missing levels and the make sure they are in a predictable order
        for factor in master.keys():
            for f in master[factor]:
                # Add the category if it is not present
                if f not in df[factor].cat.categories.values:
                    df[factor].cat.add_categories(f, inplace=True)
            # Enforce a known ordering
            df[factor].cat.reorder_categories(sorted(df[factor].cat.categories), inplace=True)
