"""This package is for managing a collection of Examples.

ExampleSet objects are typically created by a DataSet, but may be created directly.

Basic Usage Examples:

Start by saving this data in my-sample-labels.txt in the Config().label_dir directory (defaults to ./data/labels/).
::

    zone	cavity	cav#	fault	time
    1L25	4	44	Microphonics	2020/03/10 01:08:41
    2L24	5	77	Controls Fault	2020/03/10 01:42:03
    1L25	5	45	Microphonics	2020/03/10 02:50:07
    2L26	8	96	E_Quench	2020/03/10 02:58:13
    1L25	5	45	Microphonics	2020/03/10 04:55:21
    1L22	4	20	Quench_3ms	2020/03/10 05:06:13
    1L25	5	45	Microphonics	2020/03/10 07:35:32
    2L22	0	57	Multi Cav turn off	2020/03/10 07:59:49
    2L23	0	65	Multi Cav turn off	2020/03/10 07:59:56
    2L24	0	73	Multi Cav turn off	2020/03/10 08:00:03

Creating from scratch.  This assumes you have label files in Config().label_dir, and will save a CSV file to
Config().output_dir (defaults to ./processed-output/)
::
    from rfwtools.example_set import ExampleSet
    from rfwtools.example_validator import ExampleValidator
    es = ExampleSet()
    es.add_label_file_data(label_files=['my-sample-labels.txt'])
    es.get_label_file_report()
    es.remove_duplicates_and_mismatches()
    es.purge_invalid_examples(ExampleValidator())
    es.save_csv("my_example_set.csv")

Reporting and Visualization.  This assumes that you have created and saved an ExampleSet as in the example above.
::
    from rfwtools.example_set import ExampleSet
    es.load_csv("my_example_set.csv")
    es.display_frequency_barplot(x='zone', color_by='cavity_label')
    es.display_zone_label_heatmap(zones=['1L22', '1L23', '1L24', '1L25', '1L26'])

es.display_summary_label_heatmap(title='2L22 7AM Summary',
                                 query = 'zone=="2L22" & dtime < "2020-03-10 08:00:00" & dtime > "2020-03-10 07:00:00"')
"""

import datetime
import warnings
from typing import List, Tuple

import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import tzlocal
from tqdm import tqdm

from rfwtools import utils
from rfwtools.config import Config
from rfwtools.example import Example
from rfwtools.timestamp import is_datetime_in_range, TimestampMapper
from rfwtools.visualize.timeline import swarm_timeline
from rfwtools.visualize import heatmap


class ExampleSet:
    """A class for managing a collection of examples, including metadata about the collection of examples.

    This class has methods for building collections of examples from our standard label files or from the waveform
    browser webservice.  It also includes many methods for visualizing and reporting.

    Attributes:
        known_zones:
            A list of strings identifying the minimum set of zone categories to be included in the categorical.  The
            class version is the default set.  The instance version is the known to that instance.
        known_cavity_labels:
            A list of strings identifying the minimum set of cavity label categories to be included in the categorical.
            The class version is the default set.  The instance version is the known to that instance.
        known_fault_labels:
            A list of strings identifying the minimum set of fault label categories to be included in the categorical.
            The class version is the default set.  The instance version is the known to that instance.

    """

    # The expected fault levels as of Dec 2020.  New faults may appear over time, but this is a baseline.
    known_zones = ['0L04', '1L07', '1L22', '1L23', '1L24', '1L25', '1L26', '2L22', '2L23', '2L24', '2L25', '2L26']
    known_cavity_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
    known_fault_labels = ['Single Cav Turn off', 'Multi Cav turn off', 'E_Quench', 'Quench_3ms',
                            'Quench_100ms', 'Microphonics', 'Controls Fault', 'Heat Riser Choke', 'Unknown']

    # Expected column names
    __columns = ['zone', 'dtime', 'cavity_label', 'fault_label', 'cavity_conf', 'fault_conf', 'example', 'label_source']

    def __init__(self, known_zones: List[str] = None, known_cavity_labels: List[str] = None,
                 known_fault_labels: List[str] = None):
        """Create an instance of an ExampleSet.  Optionally override the default levels for zones and labels.

        Arguments:
            known_zones:
                A list of strings identifying the minimum set of zone categories to be included in the categorical.
            known_cavity_labels:
                A list of strings identifying the minimum set of cavity label categories to be included in the categorical.
            known_fault_labels:
                A list of strings identifying the minimum set of fault label categories to be included in the categorical
        """

        # Setup the standard default values for zone and label options
        if known_zones is None:
            self.known_zones = ExampleSet.known_zones
        else:
            self.known_zones = known_zones

        if known_cavity_labels is None:
            self.known_cavity_labels = ExampleSet.known_cavity_labels
        else:
            self.known_cavity_labels = known_cavity_labels

        if known_fault_labels is None:
            self.known_fault_labels = ExampleSet.known_fault_labels
        else:
            self.known_fault_labels = known_fault_labels

        # Construct an empty DataFrame with proper dtypes
        self.__example_df = pd.DataFrame(
            {'zone': pd.Categorical([], categories=self.known_zones),
             'dtime': pd.Series([], dtype='datetime64[ns]'),
             'cavity_label': pd.Categorical([], categories=self.known_cavity_labels),
             'fault_label': pd.Categorical([], categories=self.known_fault_labels),
             'cavity_conf': pd.Series([], dtype="float64"),
             'fault_conf': pd.Series([], dtype='float64'),
             'example': pd.Series([], dtype="object"),
             'label_source': pd.Series([], dtype="object")
             }
        )

        # Create a hash for holding on to the label file data.  This will preserve the original data after cleaning for
        # duplicates, mismatches, etc.
        self.label_file_dataframes = {}

    def save_csv(self, filename: str, out_dir: str = None, sep: str = ',') -> None:
        """Write out the ExampleSet data as a CSV file relative to out_dir.  Only writes out example_df equivalent.

        Arguments:
            filename: The filename to save.  Will be relative out_dir
            out_dir: The directory to save the file in.  Defaults to Config().output_dir
            sep: Delimiter string used by Pandas to parse given "csv" file
        """
        if out_dir is None:
            out_dir = Config().output_dir
        self.__example_df.drop('example', axis=1).to_csv(os.path.join(out_dir, filename), sep=sep, index=False)

    def load_csv(self, filename: str, in_dir: str = None, sep: str = ',') -> None:
        """Read in a CSV file that has ExampleSet data.

        Arguments:
            filename: The filename to save.  Will be relative in_dir
            in_dir: The directory to find the file in.  Defaults to Config().output_dir
            sep: Delimiter string used by Pandas to parse given "csv" file

        Raises:
            ValueError: If the CSV file does not have the expected column names.
        """
        if in_dir is None:
            in_dir = Config().output_dir
        if type(filename).__name__ == 'str':
            df = pd.read_csv(os.path.join(in_dir, filename), sep=sep)
        else:
            # Allows for tricks with file-like objects
            df = pd.read_csv(filename, sep=sep)

        if sorted(df.columns.to_list()) != sorted(ExampleSet.__columns[0:6] + ExampleSet.__columns[7:]):
            raise ValueError("Cannot load CSV file.  Unexpected column format.")

        # Put the DataFrame into a standard structure - categories, column order, etc.
        ExampleSet.__standardize_df_format(df)

        # Add the example column
        df['example'] = df.apply(ExampleSet.__Example_from_row, axis=1, raw=False)

        self.__example_df = df

    def update_example_set(self, df: pd.DataFrame, keep_label_file_dataframes: bool = False) -> None:
        """Replaces the contents of this ExampleSet with the supplied DataFrame.

        Note: A copy of df is used.

        Arguments:
            df: A DataFrame formatted for ExampleSet that will replace the the contents of this ExampleSet.
            keep_label_file_dataframes: Should the dictionary of label file DataFrames be kept.  If False, the
                                         dictionary recreated.  If True, no action is taken.

        Raises:
            ValueError: If columns do not match
        """

        e_df_cols = sorted(self.__example_df.columns)
        df_cols = sorted(df.columns)
        # Check that the have the same column names
        if e_df_cols != df_cols:
            raise ValueError(f"Columns do not match. {e_df_cols != df_cols}")
        # Check that all of the columns have matching dtypes.  CategoricalDtypes match only if the have the same
        # categories, not only if they are by categorical.  Here that's a problem since new categories may arise.  Just
        # check that the names of the dtypes match.
        e_df_dtype_names = [str(x) for x in self.__example_df[e_df_cols].dtypes]
        df_dtype_names = [str(x) for x in df[df_cols].dtypes]

        if not e_df_dtype_names == df_dtype_names:
            raise ValueError(f"Column dtypes do not match. {e_df_dtype_names} != {df_dtype_names}")

        if not keep_label_file_dataframes:
            self.label_file_dataframes = {}

        self.__example_df = df.copy()

    def get_example_df(self) -> pd.DataFrame:
        """Returns the example set as a DataFrame (copy)

        Returns:
            A copy of the internal ExampleSet DataFrame
        """
        return self.__example_df.copy()

    def add_label_file_data(self, label_files: List[str] = None, exclude_zones: List[str] = None,
                            exclude_times: List[Tuple[datetime.datetime, datetime.datetime]] = None) -> None:
        """Process and add label files' data to the ExampleSet's internal collection.

        Arguments:
            label_files:
                List of label files to process.  If None, all files in Config().label_dir are read.  Relative paths are
                resolved relative to Config().label_dir.
            exclude_zones:
                List of zones to exclude.  Defaults to Config().exclude_zones.
            exclude_times:
                List of 2-tuples of datetime objects.  Each 2-tuple specifies a range to exclude.  None implie +/-Inf.
        """

        # Use the defaults from the config file if None is given
        e_zones = exclude_zones if exclude_zones is not None else Config().exclude_zones
        e_times = exclude_times if exclude_times is not None else Config().exclude_times
        l_files = label_files
        if l_files is None:
            # Only want to process regular files, not directories, etc.
            l_files = [f for f in os.listdir(Config().label_dir) if os.path.isfile(os.path.join(Config().label_dir, f))]

        if len(l_files) == 0:
            raise RuntimeError(f"No label files specified or discovered in default label_dir '{Config().label_dir}'")

        # Iterate through the supplied label files.  Non-absolute paths will be assumed to be relative to the configured
        # label directory
        for label_file in l_files:
            if not os.path.isabs(label_file):
                label_file = os.path.join(Config().label_dir, label_file)

            # Process the label file into a DataFrame
            df = ExampleSet._create_dataframe_from_label_file(label_file, e_zones, e_times)

            # Stash the label file DataFrame into a dictionary in case we needed it later
            self.label_file_dataframes[label_file] = df.copy()

            # Add the DataFrame to the internal collection
            self._add_example_df(df)

    def add_web_service_data(self, server=None, begin=None, end=None, models=None):
        """Add web service data (faults labeled by in-service model) to the example set.

        Note - should be used exclusive of label data since they will largely overlap

        Arguments:
            server (str) - The server to query for the data.  If None, use the value in Config
            begin (datetime) - The earliest time for which a fault should be included.  If None, defaults to Jan 1, 2018
            end (datetime) - The latest time for which a fault should be included.  If None defaults to "now"
            models (list(str)) - A list of model names that should be included in the results.  None means include all

        Returns (None) - Returns nothing.
        """

        if server is None:
            server = Config().data_server

        if begin is None:
            begin = datetime.datetime(year=2018, month=1, day=1)

        if end is None:
            end = datetime.datetime.now()

        # Get the data from the web service
        df = ExampleSet._create_dataframe_from_web_query(server=server, begin=begin, end=end, models=models)

        # Add it to the existing ExampleSet
        self._add_example_df(df)

    def get_label_file_report(self):
        """Generate a string containing a report on the processed label files"""

        # Check to see if we have any duplicates and print them out
        num_total_events = self.count_events()
        num_total_labels = len(self.__example_df)
        num_events_with_multiple_labels = self.count_duplicated_events()
        num_duplicate_labels = self.count_duplicated_labels()
        num_events_with_mismatched_labels = self.count_duplicated_events_with_mismatched_labels()
        num_mismatched_labels = self.count_mismatched_labels()
        mismatched_output = "None Found"
        if num_events_with_mismatched_labels != 0:
            mismatched_output = self.get_events_with_mismatched_labels().to_string()

        out = f"""#### Summary ####
Note: event == unique zone/timestamp, label == row in label_file

Number of events: {num_total_events}
Number of labels: {num_total_labels}
Number of events with multiple labels: {num_events_with_multiple_labels}
Number of duplicate labels: {num_duplicate_labels}
Number of 'extra' labels: {num_duplicate_labels - num_events_with_multiple_labels}

Number of events with mismatched labels: {num_events_with_mismatched_labels}
Number of mismatched labels: {num_mismatched_labels}

#### Events With Mismatched Labels ####
{mismatched_output} 
"""

        return out

    def remove_duplicates_and_mismatches(self, report=False):
        """Removes duplicate example entries and removes all instances of examples that have mismatched labels

        Args:
            report (bool) - Should information about what was removed be included?
        """
        # Split into event groups
        gb = self.__example_df.groupby(['zone', 'dtime'])

        # Keep only groups that that have exactly one unique cavity and fault label
        df = gb.filter(lambda x: x.cavity_label.nunique() == 1 and x.fault_label.nunique() == 1)

        # Print out the entries that were removed
        if report:
            tmp_df = gb.filter(lambda x: not (x.cavity_label.nunique() == 1 and x.fault_label.nunique() == 1))
            print(f"## Removing the following {len(tmp_df)} entries from the ExampleSet as label mismatches ##")
            print(tmp_df.sort_values(["zone", "dtime"]).to_string())
            print("\n")

        # Track the size so we can report if needed
        orig_size = len(self.__example_df)

        # Replace the original example_df with this reduced set.
        self.__example_df = df.drop_duplicates(["zone", "dtime"])

        # Print out how many entries were removed as duplicates
        if report:
            num_dupes = orig_size - len(self.__example_df)
            print(f"## Removed {num_dupes} entries from the ExampleSet for being duplicates ##")

    def purge_invalid_examples(self, validator, report=True, progress=True):
        """Removes all examples from the ExampleSet that do not pass validation

        Args:
            validator (callable) - A function that accepts an Example as an argument and raises an exception if the
                                   Example is not valid
            report (bool) - Should information about what is purged be printed?
            progress (bool) - Should a progress bar be displayed
        """

        # Variable for report output
        out = "\n## Validation Results ##\n"

        # Count of how many examples were removed
        count = 0

        # Private function that allows for easy reporting
        def __apply_validator(row, _validator):
            """Applies the validator function in the context of DataFrame apply method.  Updates out, count from parent

            Args:
                row (DataFrame) - A DataFrame row containing the example to be validated.  Should contain an Example
                                  under a column named 'example'
                _validator (ExampleValidator) - Object doing the validation

            Returns:
                (bool) - True if example passed validation,  Otherwise, False.
            """

            # Allow this function to modify out and count from the parent function
            nonlocal out
            nonlocal count

            _validator.set_example(row.example)
            try:
                _validator.validate_data()
            except Exception as exc:
                count += 1
                msg = f"Invalid event - {row.example}\n  {exc}\n"
                out += msg

                # If we're debugging, we probably don't want to wait until the end to see what happened.
                if Config().debug:
                    print(msg)
                return False

            return True

        # tqdm provides a progress bar and the pd.DataFrame.progress_apply method.  This registers a new instance with
        # pandas
        if progress:
            # tqdm/pandas generate a Future warning about the Panel class.  Suppress that since I can't do anything
            # about it.  Will fix if it breaks.
            print("## Validating Examples ##")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The Panel class is removed from pandas.")
                tqdm.pandas()

                # Apply the validator to generate a bool column we can filter on
                valid = self.__example_df.progress_apply(func=__apply_validator, axis=1, _validator=validator)
        else:
            # Apply the validator to generate a bool column we can filter on
            valid = self.__example_df.apply(func=__apply_validator, axis=1, _validator=validator)

        if report:
            print(out)
            print(f"\nPurging {count} invalid examples")

        # Keep only the events that are valid
        self.__example_df = self.__example_df[valid]

    def _add_example_df(self, df, allow_new_columns=False):
        """Add a DataFrame of examples to the ExampleSet's internal collection.

        Args:
            df (DataFrame) - A dataframe of examples to be added to the existing examples
            allow_new_columns (bool) - An exception will be raised if df has any columns that do not map to existing
                                       attributes in the existing collection (e.g., you would be adding new columns to
                                       an existing DataFrame)
        """

        # Union the categories present in the existing examples with those presented in new examples
        for col in [col for col in self.__example_df.columns if self.__example_df[col].dtype.name == 'category']:
            uc = pd.api.types.union_categoricals([self.__example_df[col], df[col]])
            self.__example_df[col] = pd.Categorical(self.__example_df[col], categories=uc.categories)
            df[col] = pd.Categorical(df[col], categories=uc.categories)

        # Make sure we are adding similar data unless otherwise stated
        if not allow_new_columns:
            if (len(df.columns.values.tolist()) != len(self.__example_df.columns.values.tolist())) and (
                    sorted(df.columns.values.tolist()) != sorted(self.__example_df.columns.values.tolist())):
                raise ValueError(
                    "New DataFrame does not have same columns as example_df and allow_new_columns=False")

        # Add the new data to the bottom of the internal DataFrame, and add it to the dict of included label files
        self.__example_df = pd.concat((self.__example_df, df), ignore_index=True)

    @staticmethod
    def _create_dataframe_from_web_query(server=None, begin=None, end=None, models=None):
        """This creates a ExampleSet consistent DataFrame based on the responses of the web query.  Labeled faults only.

        Args:
            server (str) - The server to query for the data.  If None, use the value in Config
            begin (datetime) - The earliest time for which a fault should be included.  If None, defaults to Jan 1, 2018
            end (datetime) - The latest time for which a fault should be included.  If None defaults to "now"

        Returns (DataFrame) - The ExampleSet consistent DataFrame containing the web query response
        """

        # Make the web query and get results
        web_fmt = "%Y-%m-%d %H:%M:%S"
        web_events = utils.get_events_from_web(server, begin=begin.strftime(web_fmt), end=end.strftime(web_fmt))

        # Parse the web query.  The web service returns fault events with a UTC timestamp.  We convert it to the
        # localtime zone for simplicity and compatibility with the (untimezoned) label files.  Assumption here is
        # that we are running this code in the same timezone as CEBAF is in.
        # TODO - Address this timezone problem in both label file and here
        fmt = "%Y-%m-%d %H:%M:%S.%f%z"
        event_list = web_events['events']
        extracted_events = list()
        for event in event_list:
            # Get a timezone aware datetime object of UTC timestamp (manually add GMT offset string) then convert it
            # to local time
            dt_local = datetime.datetime.strptime(event['datetime_utc'] + "-00:00", fmt).astimezone(
                tzlocal.get_localzone()).replace(tzinfo=None)
            zone = event['location']

            # Read in label info
            f_label = None
            c_label = None
            f_conf = None
            c_conf = None
            l_source = None

            # Skip any fault events that were not labeled
            if event['labels'] is not None:
                for label in event['labels']:

                    # Check that this was labeled by one of the models we requested
                    if models is not None:
                        if label['model-name'] not in models:
                            continue

                    # Process the model source
                    if l_source is None:
                        l_source = label['model-name']
                    elif l_source != label['model-name']:
                        # Make the source a combo with cavity model first
                        if label['name'] == "cavity":
                            l_source = f"{label['model-name']}/{l_source}"
                        elif label['name'] == "fault-type":
                            l_source = f"{l_source}/{label['model-name']}"
                        else:
                            print(f"Skipping {zone} / {dt_local} because of unrecognized label name")
                            continue

                    # The operator facing models may present slightly processed label names.  Here we convert back to
                    # names used in the label files.  I guess this is a potential error point should future models
                    # use these names in a different way.  Not sure what to do about it here though.
                    if label['name'] == "cavity":
                        if label['value'] == 'multiple':
                            c_label = "0"
                        else:
                            c_label = label['value']
                        c_conf = label['confidence']
                    elif label['name'] == "fault-type":
                        if label['value'] == 'Multi Cav Turn off':
                            f_label = "Multi Cav turn off"
                        else:
                            f_label = label['value']
                        f_conf = label['confidence']

            # We only want labeled data
            if f_label is None or c_label is None:
                continue

            # Accumulate the events into a list of dictionaries.  Each dictionary is one event
            extracted_events.append(
                {'zone': zone, 'dtime': dt_local, 'fault_label': f_label, 'cavity_label': c_label,
                 'fault_conf': f_conf, 'cavity_conf': c_conf, 'label_source': l_source})

        # Construct an empty DataFrame
        df = pd.DataFrame({
            'zone': pd.Categorical([]),
            'dtime': pd.Series([], dtype='datetime64[ns]'),
            'fault_label': pd.Categorical([]),
            'cavity_label': pd.Categorical([]),
            'cavity_conf': pd.Series([], dtype='float64'),
            'fault_conf': pd.Series([], dtype='float64'),
            'label_source': pd.Series([], dtype='object')
        })

        # Append the fault events to the DataFrame
        for event in extracted_events:
            df = df.append(event, ignore_index=True)

        # Operates in place on DataFrame
        ExampleSet.__standardize_df_format(df)

        return df

    @staticmethod
    def _create_dataframe_from_label_file(filepath, exclude_zones=None, exclude_times=None):
        """This parses the DataSet's specified label files and saves the constructed Examples"""

        # This is the header we expect in all files - tab separated
        exp_header = "zone	cavity	cav#	fault	time\n"

        # Work through the file and build a dictionary keyed on events
        # with an array of labels found for each event.  We'll print out summary information,
        # and then print label files for each "good" event
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found - {filepath}")

        if Config().debug:
            print(f"Processing {filepath}")

        zones = []
        dts = []
        c_labels = []
        f_labels = []
        c_confs = []
        f_confs = []
        l_sources = []

        # Read each file line by line and create a new TSV file for each
        # labeled example we encounter
        with open(filepath, 'r') as fh:
            # Toss the header line by reading another one in loop - ignore any trailing whitespace
            line = fh.readline().lstrip()
            header = line

            if Config().debug:
                print("Skipping header: {}".format(header))
            if header != exp_header:
                print("Error: Unexpected header: '{}'".format(header))

            # Keep track of how many lines were skipped due to some error
            skip_count = 0

            # Process each line.  At this point we are expected labeled examples or comments
            while line:

                # Strip off leading and trailing whitespace
                line = fh.readline().rstrip().lstrip()

                # Check special cases
                if not line:
                    if Config().debug:
                        print("Found last line.")
                    break
                if line.startswith("#"):
                    if Config().debug:
                        print("Skipping: '{}'".format(line))
                    continue
                if len(line) == 0:
                    continue

                # Process the label fields
                fields = line.split('\t')
                zone = fields[0]
                cavity_label = fields[1]
                fault_label = fields[3]

                # Label files don't provide confidence levels
                cavity_conf = None
                fault_conf = None

                try:
                    tsm = TimestampMapper()
                    ts = tsm.get_full_timestamp(zone, datetime.datetime.strptime(fields[4], "%Y/%m/%d %H:%M:%S"))
                except ValueError as exc:
                    skip_count += 1
                    print("Error processing line '{}'.".format(line))
                    print(f"    {exc}")
                    continue

                # Check if the zone should be excluded
                if exclude_zones is not None and zone in exclude_zones:
                    continue

                # check if the label should be excluded because of the timestamp
                if is_datetime_in_range(ts, exclude_times):
                    continue

                # Add entries to all of the lists for this example.
                zones.append(zone)
                dts.append(ts)
                c_labels.append(cavity_label)
                f_labels.append(fault_label)
                c_confs.append(cavity_conf)
                f_confs.append(fault_conf)
                l_sources.append(os.path.basename(filepath))

                if Config().debug:
                    print("Processed: {} {} - {}".format(zone, ts, line))

        if Config().debug:
            print(f"Skipped {skip_count} events from {filepath} due to processing issues")

        # Construct a DataFrame for the new data
        df = pd.DataFrame(
            {'zone': pd.Categorical(zones),
             'dtime': dts,
             'cavity_label': pd.Categorical(c_labels),
             'fault_label': pd.Categorical(f_labels),
             'cavity_conf': c_confs,
             'fault_conf': f_confs,
             'label_source': l_sources})

        # Update the DataFrame to have a standard format (column dtypes, order, etc.)  Should add example column.
        ExampleSet.__standardize_df_format(df)

        return df

    #### Reporting-related methods ####
    def count_events(self):
        """Returns the number of unique events (zone/datetime combinations

        This would count as two since two unique zone/datetime pairs appeared
        4240  2L25 2020-09-21 06:53:16.500            5             E_Quench
        4241  2L26 2020-09-22 06:53:17.500            6             E_Quench
        4242  2L26 2020-09-22 06:53:17.500            6             E_Quench
        """
        return len(self.__example_df.drop_duplicates(subset=['zone', 'dtime']))

    def count_labels(self):
        """Returns the number of labels (rows in label files)"""
        return len(self.__example_df)

    def get_duplicated_labels(self):
        """"Returns a DataFrame containing labels for events that appear multiple times"""
        # Split on event.  observed=True only includes categorical levels that are seen and improves performance
        gb = self.__example_df.groupby(["zone", "dtime"], as_index=False, observed=True)

        # Keep event groups that have > 1 rows.  Return length of the resulting DataFrame
        return gb.filter(lambda x: len(x) > 1)

    def count_duplicated_events(self):
        """Returns the number of events that appear multiple times, i.e., were labeled more than once.

        This would count as one since only one event appeared that did occur multiple times
        4240  2L25 2020-09-21 06:53:16.500            5             E_Quench
        4241  2L26 2020-09-22 06:53:17.500            6             E_Quench
        4242  2L26 2020-09-22 06:53:17.500            6             E_Quench
        """
        # Get the duplicated labels, then remove duplicate zone/timestamp pairs
        return len(self.get_duplicated_labels().drop_duplicates(["zone", "dtime"]))

    def count_duplicated_labels(self):
        """Returns the number of labeling occurrences for events that appear multiple times"""
        return len(self.get_duplicated_labels())

    def get_unduplicated_events(self):
        """Returns a DataFrame of the events that appear exactly once in the ExampleSet"""
        # Split on event.  observed=True only includes categorical levels that are seen and improves performance
        gb = self.__example_df.groupby(["zone", "dtime"], as_index=False, observed=True)

        # Keep event groups that have exactly one row.
        return gb.filter(lambda x: len(x) == 1)

    def count_unduplicated_events(self):
        """Returns the number of events that appear exactly once.

        This would count as one since only one event appeared that did not occur multiple times
        4240  2L25 2020-09-21 06:53:16.500            5             E_Quench
        4241  2L26 2020-09-22 06:53:17.500            6             E_Quench
        4242  2L26 2020-09-22 06:53:17.500            6             E_Quench
        """

        #  Return the length of resulting DataFrame
        return len(self.get_unduplicated_events())

    def get_events_with_mismatched_labels(self):
        """Returns a DataFrame containing the events that have mismatched labels"""
        # Split on events.
        gb = self.__example_df.groupby(['zone', 'dtime'], as_index=False, observed=True)

        # Keep event groups that have more than one unique fault or cavity label. Return the length of resulting
        # DataFrame.
        return gb.filter(lambda x: x.cavity_label.nunique() > 1 or x.fault_label.nunique() > 1)

    def count_duplicated_events_with_mismatched_labels(self):
        """Returns the number of events that appear multiple times with different labels

        This would count as one since one event appeared that had mismatched labels
        4240  2L26 2020-09-21 06:53:16.500            5             E_Quench
        4241  2L26 2020-09-21 06:53:16.500            6             E_Quench
        4242  2L26 2020-09-21 06:53:16.500            6             E_Quench
        """
        # Get events that have mismatched labels
        mismatch_df = self.get_events_with_mismatched_labels()

        # Drop duplicates so that we have the event count, not the count of mismatched occurrences.  Return the length
        # of resulting DataFrame
        return len(mismatch_df.drop_duplicates(['zone', 'dtime']))

    def count_mismatched_labels(self):
        """Returns the number of times an event with mismatched labels appears in the ExampleSet.

        This would count as three mismatched labels since one event with mismatched labels appeared three times
        4240  2L26 2020-09-21 06:53:16.500            5             E_Quench
        4241  2L26 2020-09-21 06:53:16.500            6             E_Quench
        4242  2L26 2020-09-21 06:53:16.500            6             E_Quench
        """
        return len(self.get_events_with_mismatched_labels())

    #### Visualization Methods ####
    def display_timeline(self, query=None, **kwargs):
        """Display a timeline of examples as a swarmplot

        Args:
            query (str) - The expr argument to DataFrame.query.  Subsets data before plot
            kwargs (dict) - Other named parameters are passed to swarm_timeline method
        """
        df = self.__example_df.copy()
        if query is not None:
            df = df.query(query)
        swarm_timeline(df, **kwargs)

    def display_summary_label_heatmap(self, title="Label Summary", query=None):
        """Display a heatmap of fault vs cavity labels for all examples in this object

        Args:
            title (str) - The title of the plot
            query (str) - The expr argument to DataFrame.query.  Subsets data before plot
        """
        df = self.__example_df.copy()
        if query is not None:
            df = df.query(query)
        heatmap.heatmap_cavity_vs_fault_label_counts(data=df, title=title)

    def display_zone_label_heatmap(self, zones=None, query=None):
        """Display a heatmap of fault vs cavity labels for all examples in this object for each unique zone category

        Args:
            zones (list(str)) - A list of the zones to display.
            query (str) - The expr argument to DataFrame.query.  Subsets data before plot
        """

        if zones is None:
            zones = self.__example_df.zone.cat.categories

        df = self.__example_df.copy()
        if query is not None:
            df = df.query(query)

        heatmap.show_fault_cavity_count_by_zone(df, zones=zones)

    def display_examples_by_weekday_barplot(self, color_by=None, title=None, query=None):
        """Show example counts by the day of the week as a stacked barplot

        Args:
            color_by (str) - The DataFrame column on which the bars will be split/colored.
            title (str) - The title to put on the plot.  A reasonable default will be generated if None.
            query (str) - The expr argument to DataFrame.query.  Subsets data before plot
        """

        df = self.__example_df.copy()

        # Query/subset the data
        if query is not None:
            df = df.query(query)

        # Get the day names
        day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        df['day'] = df['dtime'].dt.day_name()
        df['day'] = pd.Categorical(df['day'], categories=day_names)

        if color_by is None:
            if title is None:
                title = "Example Count by Day of Week"

            # Get the counts by day
            count_df = df.groupby(['day'])['day'].count().unstack(color_by).loc[day_names, :]
        else:
            if title is None:
                title = f"{color_by} Count by Day of Week"
            df[color_by] = self.__example_df[color_by]

            # Get the counts by the color_by column
            count_df = df.groupby(['day', color_by])['day'].count().unstack(color_by).loc[day_names, :]

        # Create the plot
        ax = count_df.plot(kind="bar", stacked=True, title=title)

        # Format the legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels), loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

        # Display it
        plt.gcf().subplots_adjust(left=0.1, top=0.9, right=0.7, bottom=0.2)
        plt.show()

    def display_frequency_barplot(self, x, color_by=None, title=None, query=None):
        """Display the example count against one or two different factors, as a (stacked) bar chart.

        Args:
            x (str) - The column name for which each bar will appear.  Should probably be categorical.
            color_by (str) - The column name by which each bar will be split and colored (for a stacked bar plot).  If
                              None, then a simple bar plot will be displayed.
            title (str) - The title to put on the chart.  If None, a reasonable default will be generated.
            query (str) - The expr argument to DataFrame.query.  Subsets data before plot
        """

        df = self.__example_df.copy()
        if query is not None:
            df = df.query(query)

        # Set a reasonable default
        if title is None:
            start = df["dtime"].min().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5]
            end = df["dtime"].max().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5]
            title = f"{x}\n({start} - {end})"

        if color_by is None:
            # Simple chart if no factor to color by
            df[x].value_counts().sort_index().plot(kind="bar", title=title)

        else:
            # Get the counts
            count_df = df.groupby([x, color_by])[x].count()

            # Create the plot
            ax = count_df.unstack(color_by).fillna(0).plot(kind='bar', stacked=True, title=title)

            # Format the legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(reversed(handles), reversed(labels), loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

        # Display it
        plt.subplots_adjust(left=0.1, top=0.9, right=0.7, bottom=0.4)
        plt.show()

    def get_classification_report(self, other, label="cavity_label", query=None, other_query=None):
        """This prints a classification report of this ExampleSet's cavity labels considering other as ground truth.

        Only examples from other for which there is an example in this ExampleSet are considered

        Args:
            other (ExampleSet): An ExampleSet that contains cavity labels considered the ground truth.
            label (str): The column name of containing the label values to compare.
            query (str) - The expr argument to DataFrame.query.  Subsets data before comparison.
        """

        # Subset this ExampleSet if requested
        df = self.__example_df.copy()
        if query is not None:
            df = df.query(query)

        # Subset the other ExampleSet if requested
        o_df = other.get_example_df()
        if other_query is not None:
            o_df = o_df.query(query)

        df = df.merge(o_df[['zone', 'dtime', label]], how="inner", on=['zone', 'dtime'])
        print(classification_report(y_true=df[label + "_y"], y_pred=df[label + "_x"]))

    def __eq__(self, other):
        """Check if this ExampleSet is equivalent to the other."""

        # Short circuit check
        if self is other:
            return True

        # Short circuit check
        if type(self) != type(other):
            return False

        eq = True
        # Check the example DataFrame - first consider None case
        if self.__example_df is None and other.get_example_df() is not None:
            eq = False
        elif not self.__example_df.equals(other.get_example_df()):
            eq = False
        # Check the dict of label file dataframes - first consider None case
        elif self.label_file_dataframes is None and other.label_file_dataframes is not None:
            eq = False
        elif self.label_file_dataframes is not None and other.label_file_dataframes is None:
            eq = False
        # Now check that the contents/lengths are the same
        elif len(self.label_file_dataframes.keys()) != len(other.label_file_dataframes.keys()):
            eq = False
        elif len(self.label_file_dataframes.keys()) == len(other.label_file_dataframes.keys()):
            for k in self.label_file_dataframes.keys():
                if k not in other.label_file_dataframes.keys():
                    eq = False
                    break
                if not self.label_file_dataframes[k].equals(other.label_file_dataframes[k]):
                    eq = False
                    break

        return eq

    @staticmethod
    def __Example_from_row(x):
        """Creates an Example object from a row of a standard ExampleSet DataFrame"""
        return Example(x.zone, x.dtime, x.cavity_label, x.fault_label, x.cavity_conf, x.fault_conf, x.label_source)

    @staticmethod
    def __standardize_df_format(df):
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
        df.fault_conf = df.fault_conf.astype("float64")
        df.cavity_conf = df.cavity_conf.astype("float64")

        # Construct the Example objects based on row values if needed
        if 'example' not in df.columns.to_list():
            df['example'] = df.apply(ExampleSet.__Example_from_row, axis=1, raw=False)

        # Ensure a consistent set of category levels and their order.
        master = {
            'zone': ExampleSet.known_zones,
            'fault_label': ExampleSet.known_fault_labels,
            'cavity_label': ExampleSet.known_cavity_labels
        }

        # Add any missing levels and the make sure they are in a predictable order
        for factor in master.keys():
            for f in master[factor]:
                # Add the category if it is not present
                if f not in df[factor].cat.categories.values:
                    df[factor].cat.add_categories(f, inplace=True)
            # Enforce a known ordering
            df[factor].cat.reorder_categories(sorted(df[factor].cat.categories), inplace=True)
