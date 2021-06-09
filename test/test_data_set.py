import datetime
import math
import unittest
from unittest import TestCase

from rfwtools.config import Config
from rfwtools.data_set import DataSet
import os
import pandas as pd
import test

# Prime the pump on the timestamp map.
from rfwtools.example import ExampleType, WindowedExample
from rfwtools.feature_set import FeatureSet

test.load_timestamp_map()

# Update the config object to reflect these paths
Config().label_dir = test.test_label_dir
Config().output_dir = test.test_output_dir
Config().data_dir = os.path.join(test.test_data_dir, 'tmp')

pd.set_option("display.max_columns", None)


def my_extract(ex, my_var=None):
    """A simple extractor function for use with a DataSet"""
    if my_var is None:
        out = pd.DataFrame({"second": [ex.event_datetime.second], 'minute': [ex.event_datetime.minute]})
    else:
        out = pd.DataFrame({"second": [ex.event_datetime.second], 'minute': [ex.event_datetime.minute],
                            "dummy": [my_var]})
    return out


class TestDataSet(TestCase):
    old_config = None

    @classmethod
    def setUpClass(cls):
        cls.old_config = Config().dump_yaml_string()

    @classmethod
    def tearDownClass(cls):
        Config().load_yaml_string(cls.old_config)

    def test_save_load(self):
        # Produce a small DataSet
        Config().label_dir = test.test_label_dir
        ds = DataSet()

        # Validating these results takes ~20 seconds - optimization, I'm looking at you!
        ds.produce_example_set(progress=False)
        ds.produce_feature_set(my_extract)

        # Set a Config parameter to make sure it persists
        old_debug = Config().debug
        Config().debug = True

        # Try to save it.  Make sure there wasn't something already there that could gunk up the test
        filename = os.path.join(Config().output_dir, "test_data.pkl.bz2")
        if os.path.exists(filename):
            os.unlink(filename)

        # Save the DataSet
        ds.save(filename)

        # Change something about the config since it has some complexity compared to other elements
        Config().debug = 18

        # Load back the saved DataSet
        result = DataSet.load(filename)

        # Remove the saved file
        if os.path.exists(filename):
            os.unlink(filename)

        # Test if we got a match
        self.assertEqual(ds, result)
        self.assertEqual(True, Config().debug)

        # Set this back so we don't change it for later
        Config().debug = old_debug

    def test_produce_example_set(self):
        ds = DataSet(label_files=['test1.txt', 'test2.txt'])
        ds.produce_example_set(progress=False)

        # Check that we got something.
        self.assertIsNotNone(ds.example_set)
        self.assertIsNotNone(ds.example_set.get_example_df())
        self.assertIsNotNone(ds.example_set_model)
        self.assertIsNotNone(ds.example_set_model.get_example_df())

    def test_produce_feature_set(self):
        ds = DataSet(label_files=['test1.txt', 'test2.txt'])
        ds.produce_example_set(get_model_data=False, progress=False)
        ds.produce_feature_set(my_extract, max_workers=1)

        # Manually construct the expected feature set
        fmt = "%Y-%m-%d %H:%M:%S.%f"
        df = pd.DataFrame({
            "zone": pd.Categorical(['1L23', '1L24'], categories=['0L04', '1L07', '1L22', '1L23', '1L24', '1L25', '1L26',
                                                                 '2L22', '2L23', '2L24', '2L25', '2L26']),
            'dtime': [datetime.datetime.strptime('2020-01-08 09:12:53.3', fmt),
                      datetime.datetime.strptime('2020-01-08 09:13:00.6', fmt)],
            'cavity_label': pd.Categorical(["0", "1"], categories=['0', '1', '2', '3', '4', '5', '6', '7', '8']),
            'fault_label': pd.Categorical(['Multi Cav turn off', 'Single Cav Turn off'],
                                          categories=['Single Cav Turn off', 'Multi Cav turn off', 'E_Quench',
                                                      'Quench_3ms', 'Quench_100ms', 'Microphonics', 'Controls Fault',
                                                      'Heat Riser Choke', 'Unknown']),
            'cavity_conf': [math.nan, math.nan],
            'fault_conf': [math.nan, math.nan],
            'label_source': ['test1.txt', 'test1.txt'],
            'second': [53, 0],
            "minute": [12, 13]
        })
        exp = FeatureSet(df=df)
        pd._testing.assert_frame_equal(df.sort_index(axis=1), ds.feature_set.get_example_df().sort_index(axis=1))
        self.assertEqual(exp, ds.feature_set)

        # Test with function key word arguments
        ds.produce_feature_set(my_extract, max_workers=1, my_var="filler_feature")

        # Add in the dummy column that will be produced by including my_var
        df["dummy"] = "filler_feature"
        exp = FeatureSet(df=df)
        self.assertEqual(exp, ds.feature_set)

    def test_save_load_example_set(self):
        ds = DataSet(label_files=['test1.txt', 'test2.txt'])
        ds.produce_example_set(get_model_data=False, report=False, progress=False)

        # Grab a copy of the example set
        exp = ds.example_set

        filename = "test_example_set.pkl.bz2"
        f_path = os.path.join(Config().output_dir, filename)

        # Delete the save file if it exists
        if os.path.isfile(f_path):
            os.unlink(f_path)

        # Save ds.example_set, overwrite it, then load it
        ds.save_example_set(filename=filename)
        ds.example_set = None
        ds.load_example_set(filename=filename)

        # Delete the save file
        if os.path.isfile(f_path):
            os.unlink(f_path)

        # Compare results
        self.assertEqual(exp, ds.example_set)

    def test_save_load_windowed_example_set(self):
        ds = DataSet(label_files=['test1.txt', 'test2.txt'])
        ds.produce_example_set(get_model_data=False, report=False, progress=False)

        # Grab a copy of the example set
        exp = ds.example_set

        filename = "test_windowed_example_set.csv"
        f_path = os.path.join(Config().output_dir, filename)

        # Delete the save file if it exists
        if os.path.isfile(f_path):
            os.unlink(f_path)

        # Save ds.example_set, overwrite it, then load it
        ds.save_example_set_csv(filename=filename)
        ds.example_set = None
        ds.load_example_set_csv(filename=filename, e_type=ExampleType.WINDOWED_EXAMPLE,
                                example_kwargs={'start': -100.0, 'n_samples': 5})

        ex = ds.example_set.get_example_df().loc[0, 'example']
        self.assertTrue(isinstance(ex, WindowedExample))
        self.assertEqual(ex.n_samples, 5)
        self.assertEqual(ex.start, -100.0)

        # Delete the save file
        if os.path.isfile(f_path):
            os.unlink(f_path)

    def test_save_load_feature_set(self):
        ds = DataSet(label_files=['test1.txt', 'test2.txt'])
        ds.produce_example_set(get_model_data=False, report=False, progress=False)
        ds.produce_feature_set(my_extract, max_workers=1, my_var="filler")

        # Grab a copy of the example set
        exp = ds.feature_set

        filename = "test_feature_set.pkl.bz2"
        f_path = os.path.join(Config().output_dir, filename)

        # Delete the save file if it exists
        if os.path.isfile(f_path):
            os.unlink(f_path)

        # Save ds.example_set, overwrite it, then load it
        ds.save_feature_set(filename=filename)
        ds.feature_set = None
        ds.load_feature_set(filename=filename)

        # Delete the save file
        if os.path.isfile(f_path):
            os.unlink(f_path)

        # Compare results - these are DataFrames, so use their equals method
        self.assertEqual(exp, ds.feature_set)


if __name__ == '__main__':
    unittest.main()
