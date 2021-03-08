"""A module that manages the high level workflow from label files to FeatureSet

This module contains a single class DataSet.  It is intended to manage a standard workflow from reading and parsing
label files to generating a validated set of examples (ExampleSet) to creating a set of features that represent
the examples for model building purposes (FeatureSet).

  Typical usage example:

  ::

    ds = DataSet()
    ds.produce_example_set()
    ds.produce_feature_set(my_feature_extract_func)
    ds.save_example_set_csv("my_examples.csv")
    ds.save_feature_set_csv("my_features.csv")

"""

import bz2
import logging
import multiprocessing
import os
import pickle
import numpy as np
import pandas as pd

from typing import List, Optional

from rfwtools.example_validator import ExampleValidator
from rfwtools.example_set import ExampleSet
from rfwtools.config import Config
from rfwtools.feature_set import FeatureSet


class DataSet:
    """This class manages a standard workflow from label files on disk to the generation of features

    Attributes:
        label_files:
            A list of strings that are the label filenames in Config().label_dir.  None implies all files.
        example_validator:
            An ExampleValidator object.  This defines the characteristics of valid objects in DataSet's ExampleSet
        config_yaml:
            A YAML formatted string representing the Config() object at the time a DataSet runs save()
        feature_set:
            A FeatureSet object containing the feature data generated by produce_feature_set()
        example_set:
            An ExampleSet object that contains all of the examples generated from label_files by produce_example_set()
        example_set_model:
            An ExampleSet object containing examples labeled by the production classifier during the same time span as
            example_set.


    """

    def __init__(self, label_files: List[str] = None, example_validator: ExampleValidator = ExampleValidator()):
        """Create a DataSet instance that will collect events based on the provided label files and configured filters.

        Some filters such as excluded zones or excluded times are set in the Config objects.

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

    def produce_example_set(self, report: bool = False, progress: bool = True, get_model_data: bool = True) -> None:
        """This method causes the DataSet object to produce a set of uniquely labeled examples based on label files.

        self.example_set will contain the resulting ExampleSet.
        self.example_set_model will contain the ExampleSet corresponding to the production model's results

        Args:
            report: Whether a report detailing issues with the label file contents should be printed
            progress: Should a progress bar be displayed during validation
            get_model_data: Should the model results be queried and stored in example_set_model

        Returns:
            None
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

    def get_example_array(self) -> np.ndarray:
        """Convenience function for getting an array containing the Example objects held in self.example_set
        
        Returns:
            A numpy.ndarray generated by pd.DataFrames.values
        """
        return self.example_set.get_example_df()['example'].values

    def produce_feature_set(self, extraction_function: callable, max_workers: int = 4, verbose: bool = False,
                            **kwargs) -> None:
        """This method produces a FeatureSet by applying extraction_function to each Example in the example set.

        extraction_function should a callable that requires exactly one positional argument, the Example on which to
        operate and returns a pandas DataFrame with one row and a constant number of columns.  The extractor function
        should take care to produce identical column name values as the output from each Example is appended to a
        master DataFrame. The extraction function is responsible for loading/unloading an Example's data as needed.
        Additional kwargs will be passed to this function.  A return value of None will cause the Example to be
        excluded from the feature set.

        Additional keyword arguments will be passed to the extraction function.

        Each row represents the results of a single extraction.  A row also contains the cavity label, fault label,
        zone, timestamp, and label_source of the example.

        Note this uses python multithreading to allow for concurrent data loading/unloading.

        Args:
            extraction_function: Function that will perform feature extraction on a single Example
            max_workers: The number of parallel process workers to launch
            verbose: Controls level of print output
            **kwargs:  All additional keyword arguments will be passed to the extraction_function
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

        # ignore_index lets the new DataFrame have a useful index instead of all ones
        self.feature_set = FeatureSet(df=df)

    @staticmethod
    def _do_feature_extraction_multiprocessing(args: List) -> Optional[pd.DataFrame]:
        """Perform the feature extraction on single example.  Does NOT handle (un)loading data.

        For use with multiprocessing.  Returns a DataFrame with some metadata and the features, or None if the
        extraction_function returns None.  If an exception is raised during extraction, None is returned

        Args:
            args: A list containing (extraction_function example, verbose, kwargs) from produce_feature_set()

        Returns:
            The DataFrame produced by extraction_function with additional metadata.  Should be a single row.
        """

        (extraction_function, example, verbose, kwargs) = args
        try:
            example.load_data(verbose)

            df = pd.DataFrame({'zone': [example.event_zone],
                               'dtime': [example.event_datetime],
                               'cavity_label': [example.cavity_label],
                               'fault_label': [example.fault_label],
                               'cavity_conf': [example.cavity_conf],
                               'fault_conf': [example.fault_conf],
                               'example': [example],
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

    def save(self, filename: str, out_dir: str = None) -> None:
        """DEPRECATED Saves a bzipped pickle file containing this DataSet object.

        This is a convenient way to save all of your work within a DataSet object.  However, **changes to the installed
        rfwtools software may make it difficult or impossible to load this object** in the future.  Please save
        individual ExampleSets and FeatureSets using their supplied method (save_csv) or the DataSet wrapper method,
        e.g., save_example_set_csv().

        Note: this also saves a serialized version of the Config object.

        Args:
            filename: Filename of the save file.  Should include the .pkl.bz2 extension for clarity.
            out_dir: The directory where the file should be placed.  If None, the Config().output_dir is used.
        """
        if out_dir is None:
            out_dir = Config().output_dir

        # Cache the config since it does not pickle properly
        self.config_yaml = Config().dump_yaml_string()

        with bz2.open(os.path.join(out_dir, filename), mode="wb") as f:
            f.write(pickle.dumps(self))

    @staticmethod
    def load(filename: str, in_dir: str = None) -> 'DataSet':
        """DEPRECATED Loads a bzipped pickle file produced by save().  Returns the resulting DataSet.

        Save files made with different versions of rfwtools may be impossible to load.  Please save/load ExampleSets
        and FeatureSets using their built in CSV methods for long-term storage.

        Args:
            filename: Filename of the save file.  Should include the .pkl.bz2 extension for clarity.
            in_dir: The directory where the file should be found.  If None, the Config().output_dir is used.
        """
        if in_dir is None:
            in_dir = Config().output_dir

        with bz2.open(os.path.join(in_dir, filename), mode="rb") as f:
            ds = pickle.loads(f.read())

            # Config object does not pickle properly.  Load the cached config string
            Config().load_yaml_string(ds.config_yaml)
            return ds

    def save_feature_set(self, filename: str, out_dir: str = None) -> None:
        """DEPRECATED.  Saves a binary pickle file containing the feature set DataFrame."""
        if out_dir is None:
            out_dir = Config().output_dir

        with bz2.open(os.path.join(out_dir, filename), mode="wb") as f:
            f.write(pickle.dumps(self.feature_set))

    def load_feature_set(self, filename: str, in_dir: str = None) -> None:
        """DEPRECATED.  Loads a binary pickle file containing a feature set."""
        if in_dir is None:
            in_dir = Config().output_dir
        with bz2.open(os.path.join(in_dir, filename), mode="rb") as f:
            self.feature_set = pickle.loads(f.read())

    def save_feature_set_csv(self, filename: str, **kwargs) -> None:
        """Save a FeatureSet CSV file.  All keyword args are passed to FeatureSet.save_csv method.

        Args:
            filename: The name of the file to load.  Relative to out_dir (if supplied) or Config().output_dir.
            **kwargs: All keyword args are passed on to FeatureSet.save_csv()

        """
        self.feature_set.save_csv(filename=filename, **kwargs)

    def load_feature_set_csv(self, filename: str, **kwargs) -> None:
        """Load a FeatureSet CSV file.  Overwrites existing feature_set with new FeatureSet.

        Args:
            filename: The name of the file to load.  Relative to in_dir (if supplied) or Config().output_dir.
            **kwargs: All keyword args are passed on to FeatureSet.save_csv()
        """
        fs = FeatureSet()
        fs.load_csv(filename=filename, **kwargs)
        self.feature_set = fs

    def save_example_set(self, filename: str, out_dir: str = None) -> None:
        """DEPRECATED.  Save a binary bzipped pickle file of the current example set."""
        if out_dir is None:
            out_dir = Config().output_dir
        with bz2.open(os.path.join(out_dir, filename), mode="wb") as f:
            f.write(pickle.dumps(self.example_set))

    def load_example_set(self, filename: str, in_dir: str = None):
        """DEPRECATED.  Loads a binary bzipped pickle file containing an example set.  Old versions used pickle files"""
        if in_dir is None:
            in_dir = Config().output_dir
        with bz2.open(os.path.join(in_dir, filename), mode="rb") as f:
            self.example_set = pickle.loads(f.read())

    def save_example_set_csv(self, filename: str, **kwargs) -> None:
        """Save an ExampleSet CSV file.  All keyword args are passed to ExampleSet.save_csv method.

        Args:
            filename: The name of the file to load.  Relative to in_dir (if supplied) or Config().output_dir.
            **kwargs: All keyword args are passed on to ExampleSet.save_csv()

        """
        self.example_set.save_csv(filename=filename, **kwargs)

    def load_example_set_csv(self, filename, **kwargs):
        """Load an ExampleSet CSV file.  Overwrites existing example_set with new ExampleSet.

        Args:
            filename: The name of the file to load.  Relative to in_dir (if supplied) or Config().output_dir.
            **kwargs: All keyword args are passed on to FeatureSet.save_csv()
        """
        es = ExampleSet()
        es.load_csv(filename=filename, **kwargs)
        self.example_set = es

    def __eq__(self, other: 'DataSet') -> bool:
        """Compares DataSet objects by their example_set and feature_set parameters

        Args:
            other: The other DataSet to compare against
        """

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

    def __ne__(self, other: 'DataSet') -> bool:
        """Simple inverse of __eq__"""
        return not self.__eq__(other)
