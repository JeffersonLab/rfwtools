#import unittest
#import test_module
import os
import pickle

from rfwtools.timestamp import TimestampMapper

#def my_module_suite():
#    loader = unittest.TestLoader()
#    suite = loader.loadTestsFromModule(test_module)
#    return suite

# Define the testing locations
test_dir = os.path.dirname(os.path.realpath(__file__))
test_label_dir = os.path.join(test_dir, "test-labels")
test_output_dir = os.path.join(test_dir, 'test-output')
test_data_dir = os.path.join(test_dir, 'test-data')
tmp_data_dir = os.path.join(test_dir, 'test-data', 'tmp')


def load_timestamp_map(filename=os.path.join(test_output_dir, "ts_map.pkl")):
    """Reads the timestamp map from file if it is cached.  If not, triggers the DataSet to download it and caches it.

    Without this convenience method every call to process_label_files will download it's own map which can take around
    10 seconds.  With it, just the first one takes that long and subsequent calls take ~10ms.
    """
    ts = TimestampMapper()

    if os.path.exists(filename):
        with open(filename, "rb") as f:
            ts.update_timestamp_map(pickle.load(f))
    else:
        ts.update_timestamp_map()
        ts.save_mapper(filename)

