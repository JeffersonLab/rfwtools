import bz2
import logging
import multiprocessing
import os
import pickle
import pandas as pd

# Needed if we want to enable threading in place of multiprocessing
# import concurrent.futures
# import threading
from rfwtools.example_validator import ExampleValidator
from rfwtools.example_set import ExampleSet
from rfwtools.config import Config
from rfwtools.feature_set import FeatureSet


class DataSet:
    """This class represents the collection of events associated with an analysis"""

    def __init__(self, label_files=None, example_validator=ExampleValidator()):
        """Create a DataSet instance that will collect events based on the provided label files, filter lists.

        Args:

            label_files (list(str)) - Filenames of the label files to parse.  None means use all Config().label_dir
            example_validator (ExampleValidator) - Used to check if individual Examples from label files are valid.  No
                                                   validation performed if None
        """

        self.label_files = label_files
        self.example_validator = example_validator

        # Include this here so that saving a DataSet will also save the current Config options.
        self.config_yaml = None

        # DataFrame of extracted features with cavity and fault labels
        self.feature_set = None

        # A ExampleSet object - contains logic for managing a cohesive set of labeled fault events (i.e., examples)
        # This will hold the labels processed from label files unless otherwise modified
        self.example_set = ExampleSet()

        # An ExampleSet for holding the corresponding results from our in service models.
        self.example_set_model = ExampleSet()

    def produce_example_set(self, report=False, progress=True, get_model_data=True):
        """This method causes the DataSet object to produce a set of uniquely labeled examples based on label files.

        Args:
            report (bool): Whether a report detailing issues with the label file contents should be printed
            progress (bol): Should a progress bar be displayed during validation
            get_model_data (bool): Should the model results be queried and stored in example_set_model
        """

        # First process all of the examples from the SME label files
        self.example_set.add_label_file_data(label_files=self.label_files)
        if report:
            print(self.example_set.get_label_file_report())

        if self.example_set is None:
            self.example_set = ExampleSet()

        # Drop duplicate entries and events that have mismatched labels
        self.example_set.remove_duplicates_and_mismatches(report=report)

        # If an ExampleValidator has been specified, use it to remove any Examples that fail validation
        if self.example_validator is not None:
            self.example_set.purge_invalid_examples(self.example_validator, report=report, progress=progress)

        # Go get the model data?
        if get_model_data:
            if self.example_set_model is None:
                self.example_set_model = ExampleSet()

            # Query the web service for the corresponding date range of fault events and generate an ExampleSet
            dt = self.example_set.get_example_df()['dtime']
            self.example_set_model.add_web_service_data(begin=dt.min(), end=dt.max())

    def get_example_array(self):
        """Returns an array containing the Example objects held in self.example_set"""
        return self.example_set.get_example_df()['example'].values

    def produce_feature_set(self, extraction_function, max_workers=4, verbose=False, **kwargs):
        """This method produces a feature set by applying extraction_function to each Example in the example set.

        extraction_function should a callable that requires exactly one argument, the Example on which to operate and
        returns a pandas DataFrame with one row and a constant number of columns.  The extractor function should take
        care to produce identical column name values as the output from each Example is appended to a master DataFrame.
        The extraction function is responsible for loading/unloading an Example's data as needed.  Additional kwargs
        will be passed to this function.  A return value of None will cause the Example to be excluded from the feature
        set.

        Each row represents the results of a single extraction.  A row also contains the cavity label, fault label,
        zone, timestamp, and label_source of the example.

        Note this uses python threading to allow for concurrent data loading/unloading.  Python threading is limited
        to a single processor by the Global Interpreter Lock (GIL, at least in CPython).  This may not help if feature
        extraction is very processor centric.

        Args:
            extraction_function (callable) - The function that will perform feature extraction on a single example
            max_workers (int) - The number of parallel process workers to launch
            verbose (bool) - Controls level of print output
        """

        ##################################################
        # Concurrency using multiprocessing
        ##################################################
        if verbose:
            multiprocessing.log_to_stderr()
            logger = multiprocessing.get_logger()
            logger.setLevel(logging.INFO)

        examples = list(self.get_example_array())
        n = len(examples)

        args = list(zip([extraction_function] * n, examples, [verbose] * n, [kwargs] * n))

        with multiprocessing.Pool(max_workers) as p:
            feature_list = p.map(DataSet._do_feature_extraction_multiprocessing, args)

        # Make a DataFrame and ensure we maintain the same categorical series
        df = pd.concat(feature_list, ignore_index=True)
        for col in ('zone', 'fault_label', 'cavity_label'):
            df[col] = pd.Categorical(df[col], categories=self.example_set.get_example_df()[col].cat.categories)

        ##################################################
        # Concurrency using threading
        ##################################################
        # feature_list = list()
        # feature_list_lock = threading.Lock()
        # n = len(self.example_set)
        #
        # # Create a thread pool to do the extractions.  Doesn't parallelize across CPUs, but may improve disk IO.
        # with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        #     # map works to the shortest list, so need to manually create repeats
        #     executor.map(DataSet._do_feature_extraction_threading, [extraction_function] * n,
        #                  [self.example_set.values(), feature_list] * n, [feature_list_lock] * n)

        # ignore_index lets the new DataFrame have a useful index instead of all ones
        self.feature_set = FeatureSet(df=df)

    @staticmethod
    def _do_feature_extraction_multiprocessing(args):
        """Perform the feature extraction on single example.  Does NOT handle (un)loading data.

        For use with multiprocessing.  Returns a DataFrame with some metadata and the features, or None if the
        extraction_function returns None.  If an exception is raised during extraction, None is returned

        args - a list containing (extraction_function example, verbose, kwargs)
        """

        (extraction_function, example, verbose, kwargs) = args
        try:
            example.load_data(verbose)

            df = pd.DataFrame({'zone': [example.event_zone],
                               'dtime': [example.event_datetime],
                               'cavity_label': [example.cavity_label],
                               'fault_label': [example.fault_label],
                               'label_source': [example.label_source]})
            result = extraction_function(example, **kwargs)

            # Combine them into a single long row, with the standard info first.
            if result is not None:
                result = pd.concat((df, result), axis=1)
        except Exception as exc:
            print("### Error extracting features from event {}".format(str(example)))
            print(f"{str(exc)}")
            return None
        finally:
            example.unload_data(verbose)

        return result

    @staticmethod
    def _do_feature_extraction_threading(extraction_function, example, feature_list, list_lock):
        """Perform the feature extraction on single example.  Handles (un)loading data.

        For use with threading.
        """
        try:
            example.load_data()
            result = extraction_function(example)
            result['cavity-label'] = example.cavity_label
            result['fault-label'] = example.fault_label
            result['zone'] = example.event_zone
            result['timestamp'] = example.get_normal_time_string()
            with list_lock:
                feature_list.append(result)
        except Exception as exc:
            print("Error extracting features from event {}".format(str(example)))
            print(str(exc))
        finally:
            example.unload_data()

    def save(self, filename, out_dir=None):
        """Saves a bzipped pickle file containing this DataSet object"""
        if out_dir is None:
            out_dir = Config().output_dir

        # Cache the config since it does not pickle properly
        self.config_yaml = Config().dump_yaml_string()

        with bz2.open(os.path.join(out_dir, filename), mode="wb") as f:
            f.write(pickle.dumps(self))

    @staticmethod
    def load(filename, in_dir=None):
        """Loads a bzipped pickle file containing a DataSet object and returns the results"""
        if in_dir is None:
            in_dir = Config().output_dir

        with bz2.open(os.path.join(in_dir, filename), mode="rb") as f:
            ds = pickle.loads(f.read())

            # Config object does not pickle properly.  Load the cached config string
            Config().load_yaml_string(ds.config_yaml)
            return ds

    def save_feature_set(self, filename, out_dir=None):
        """DEPRECATED.  Saves a binary pickle file containing the feature set DataFrame."""
        if out_dir is None:
            out_dir = Config().output_dir

        with bz2.open(os.path.join(out_dir, filename), mode="wb") as f:
            f.write(pickle.dumps(self.feature_set))

    def load_feature_set(self, filename, in_dir=None):
        """DEPRECATED.  Loads a binary pickle file containing a feature set."""
        if in_dir is None:
            in_dir = Config().output_dir
        with bz2.open(os.path.join(in_dir, filename), mode="rb") as f:
            self.feature_set = pickle.loads(f.read())

    def save_feature_set_csv(self, filename, **kwargs):
        """Save a FeatureSet CSV file.  All keyword args are passed to FeatureSet.save_csv method."""
        self.feature_set.save_csv(filename=filename, **kwargs)

    def load_feature_set_csv(self, filename, **kwargs):
        """Load a FeatureSet CSV file.  Overwrites existing feature_set with new FeatureSet."""
        fs = FeatureSet()
        fs.load_csv(filename=filename, **kwargs)
        self.feature_set = fs

    def save_example_set(self, filename, out_dir=None):
        """DEPRECATED.  Save a binary bzipped pickle file of the current example set."""
        if out_dir is None:
            out_dir = Config().output_dir
        with bz2.open(os.path.join(out_dir, filename), mode="wb") as f:
            f.write(pickle.dumps(self.example_set))

    def load_example_set(self, filename, in_dir=None):
        """DEPRECATED.  Loads a binary bzipped pickle file containing an example set.  Old versions used pickle files"""
        if in_dir is None:
            in_dir = Config().output_dir
        with bz2.open(os.path.join(in_dir, filename), mode="rb") as f:
            self.example_set = pickle.loads(f.read())

    def save_example_set_csv(self, filename, **kwargs):
        """Save an ExampleSet CSV file.  All keyword args are passed to ExampleSet.save_csv method."""
        self.example_set.save_csv(filename=filename, **kwargs)

    def load_example_set_csv(self, filename, **kwargs):
        """Load an ExampleSet CSV file.  Overwrites existing example_set with new ExampleSet."""
        es = ExampleSet()
        es.load_csv(filename=filename, **kwargs)
        self.example_set = es

    def __eq__(self, other):
        """Compares DataSet objects by their label_file_events, example_sets, and feature_sets"""

        # Short circuit check.
        if self is other:
            return True

        eq = True

        if not self.example_set == other.example_set:
            eq = False
        elif self.feature_set is None and other.feature_set is not None:
            eq = False
        elif self.feature_set != other.feature_set:
            eq = False

        return eq

    def __ne__(self, other):
        """Simple inverse of __eq__"""
        return not self.__eq__(other)
