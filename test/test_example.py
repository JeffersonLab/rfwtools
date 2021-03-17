import datetime
import math
import os
import unittest

import pandas as pd
import test
from unittest import TestCase
from rfwtools.example import Example, WindowedExample, Factory, ExampleType
from rfwtools.config import Config

# A known good example available via the web service
zone = '1L22'
ts_fmt = "%m/%d/%Y, %H:%M:%S"
dt = datetime.datetime.strptime("02/24/2019, 04:22:01", ts_fmt)
cav_label = "4"
f_label = "Heat_Riser_Choke"

# Prime the pump on the timestamp map.
test.load_timestamp_map()

# Update the config object to reflect these paths
Config().label_dir = test.test_label_dir
Config().output_dir = test.test_output_dir
Config().data_dir = test.tmp_data_dir


def check_list_equal(list1, list2):
    return len(list1) == len(list2) and sorted(list1) == sorted(list2)


class TestExample(TestCase):
    old_config = None

    @classmethod
    def setUpClass(cls):
        """These tests mess with the config a bunch.  Cache the original state"""
        cls.old_config = Config().dump_yaml_string()

    @classmethod
    def tearDownClass(cls):
        """These tests mess with the config a bunch.  Return the original state"""
        Config().load_yaml_string(cls.old_config)

    def test_parse_event_dir(self):
        # Example of a simplified event.  Allows for easier checking that the data parsing is correct
        # simple_event_path = os.path.join(os.path.dirname(__file__), "test-data", "short-test")

        exp_df = pd.DataFrame({"Time": [0.1, 0.2, 0.3],
                               "1_wf1": [1, 4, 18], "1_wf2": [36, -17, 5],
                               "2_wf1": [21, 24, 218], "2_wf2": [236, -217, 25],
                               "3_wf1": [31, 34, 318], "3_wf2": [336, -317, 35],
                               "4_wf1": [41, 44, 418], "4_wf2": [436, -417, 45],
                               "5_wf1": [51, 54, 518], "5_wf2": [536, -517, 55],
                               "6_wf1": [61, 64, 618], "6_wf2": [636, -617, 65],
                               "7_wf1": [71, 74, 718], "7_wf2": [736, -717, 75],
                               "8_wf1": [81, 84, 818], "8_wf2": [836, -817, 85]
                               }, dtype='float64')

        Config().data_dir = os.path.join(test.test_data_dir, "short-test")
        simple_example = Example(zone="1L24",
                                 dt=datetime.datetime.strptime("2000_01_01 000001.1", "%Y_%m_%d %H%M%S.%f"),
                                 cavity_label=None, fault_label=None, cavity_conf=None, fault_conf=None,
                                 label_source="test")
        result_df = Example.parse_event_dir(simple_example.get_event_path())

        self.assertTrue(exp_df.equals(result_df))
        self.assertTrue(check_list_equal(exp_df.columns, result_df.columns))

    def test_load_data_flip_time_column(self):
        exp_df = pd.DataFrame({"Time": [-0.3, -0.2, -0.1],
                               "1_wf1": [1, 4, 18], "1_wf2": [36, -17, 5],
                               "2_wf1": [21, 24, 218], "2_wf2": [236, -217, 25],
                               "3_wf1": [31, 34, 318], "3_wf2": [336, -317, 35],
                               "4_wf1": [41, 44, 418], "4_wf2": [436, -417, 45],
                               "5_wf1": [51, 54, 518], "5_wf2": [536, -517, 55],
                               "6_wf1": [61, 64, 618], "6_wf2": [636, -617, 65],
                               "7_wf1": [71, 74, 718], "7_wf2": [736, -717, 75],
                               "8_wf1": [81, 84, 818], "8_wf2": [836, -817, 85]
                               }, dtype='float64')

        simple_example = Example(zone="1L24",
                                 dt=datetime.datetime.strptime("2000_01_01 000001.1", "%Y_%m_%d %H%M%S.%f"),
                                 cavity_label=None, fault_label=None, cavity_conf=None, fault_conf=None,
                                 label_source="test", data_dir=os.path.join(test.test_data_dir, "short-test"))
        simple_example.load_data()
        result_df = simple_example.event_df

        self.assertTrue(exp_df.equals(result_df))
        self.assertTrue(check_list_equal(exp_df.columns, result_df.columns))

    def test_parse_event_dir_compressed(self):
        # Example of a compressed event.

        # Make sure to include th 0..  Otherwise this ends up as an int64, not a float64
        exp_df = pd.DataFrame({"Time": [-102.4, -102.2],
                               "7_IMES": [1., 1],
                               "7_QMES": [-3., 0],
                               "7_GMES": [0., 0],
                               "8_IMES": [-4., -2],
                               "8_QMES": [-4., 0],
                               "8_GMES": [0., 0]
                               })

        compressed_example = Example(zone="1L22",
                                     dt=datetime.datetime.strptime("2019_01_30 120349.5", "%Y_%m_%d %H%M%S.%f"),
                                     cavity_label=None, fault_label=None, cavity_conf=None, fault_conf=None,
                                     label_source="test",
                                     data_dir=os.path.join(test.test_data_dir, "compressed-example"))
        compress_res = Example.parse_event_dir(compressed_example.get_event_path(compressed=True), compressed=True)
        uncompressed_res = Example.parse_event_dir(compressed_example.get_event_path(compressed=False),
                                                   compressed=False)

        # assert_frame_equal will raise if there is a problem and print out an error message
        pd._testing.assert_frame_equal(exp_df, compress_res)
        pd._testing.assert_frame_equal(compress_res, uncompressed_res)

    def test_example_construction(self):
        # Make an example object - check that this doesn't raise exception
        Config().data_dir = test.tmp_data_dir
        Example(zone=zone, dt=dt, cavity_label=cav_label, fault_label=f_label, cavity_conf=None, fault_conf=None,
                label_source="test")

    def test_example_string_ops(self):
        # Make an example object
        Config().data_dir = test.tmp_data_dir
        e = Example(zone=zone, dt=dt, cavity_label=cav_label, fault_label=f_label, cavity_conf=None, fault_conf=None,
                    label_source="test")
        # Test that the string operations are working as expected
        self.assertEqual(e._get_file_system_time_string(), "2019_02_24/042201.0")
        self.assertTupleEqual(e._get_web_time_strings(), ("2019-02-24 04:22:01", "2019-02-24 04:22:02"))

    def test_example_data_ops(self):
        Config().data_dir = test.tmp_data_dir
        e = Example(zone=zone, dt=dt, cavity_label=cav_label, fault_label=f_label, cavity_conf=None, fault_conf=None,
                    label_source="test")

        # Check that the web api download is working as expected
        self.assertIsNone(e.event_df)  # Starts out as none
        t1 = datetime.datetime.now()
        e._download_waveforms_from_web()
        t2 = datetime.datetime.now()
        if (t2 - t1).total_seconds() > 10:
            print(
                f"WARNING: download_waveforms_from_accweb took {(t2 - t1).total_seconds()} seconds (>10, includes save)")
        self.check_data_on_disk(e)

        # Try to read in the cached copy
        e.event_df = None
        self.assertIsNone(e.event_df)  # Probably overkill, but I want it to fail here if so
        t1 = datetime.datetime.now()
        e._retrieve_event_df()  # Will download data if not present, but we know it's downloaded now.
        t2 = datetime.datetime.now()
        self.check_event_df(e)
        if (t2 - t1).total_seconds() > 1:
            print(f"WARNING: retrieve_event_df_from_disk took {(t2 - t1).total_seconds()} seconds (> 1)")

        # Check that the delete works ok
        e.remove_event_df_from_disk()
        self.assertFalse(os.path.exists(e.get_event_path()))  # Was the event_directory deleted?
        e.remove_event_df_from_disk()  # This should work fine and not do anything on repeat calls

        # Check that the high level command works
        e.event_df = None
        self.assertIsNone(e.event_df)
        e._retrieve_event_df()  # Should do a second web download
        self.check_event_df(e)
        self.check_data_on_disk(e)

        # Delete the cached data to put everything back to normal
        e.remove_event_df_from_disk()

    def test_windowed_example(self):
        start = -100
        n_samples = 200 * 5  # ~0.2 ms step sizes.  I want 200 ms worth of a window here.
        end = start + (n_samples / 5)
        wex = WindowedExample(zone=zone, dt=dt, cavity_label=cav_label, fault_label=f_label, cavity_conf=math.nan,
                              fault_conf=math.nan, label_source="test", start=start, n_samples=n_samples)
        wex.load_data()

        # Check that we got something after load
        self.assertIsNotNone(wex.event_df)

        t_min = wex.event_df.Time.min()
        t_max = wex.event_df.Time.max()
        self.assertAlmostEqual(0, t_min, 1)

        # There's rounding error here so I know they should only match up to the whole number
        self.assertEqual(math.ceil(end - start), math.ceil(t_max), 1)

        # Check that we got some data
        self.assertTrue(len(wex.event_df) > 10, "wex.event_df is unexpectedly small")

        # Check that unloading works
        wex.unload_data()
        self.assertIsNone(wex.event_df, "unload_data failed to clear event_df")

        # Check that load_data() raises if it can't provide the window you're looking for.  Starts too late
        wex = WindowedExample(zone=zone, dt=dt, cavity_label=cav_label, fault_label=f_label, cavity_conf=math.nan,
                              fault_conf=math.nan, label_source="test", start=1000, n_samples=(1000 * 5))
        with self.assertRaises(RuntimeError):
            wex.load_data()

        # Not enough samples after start
        wex = WindowedExample(zone=zone, dt=dt, cavity_label=cav_label, fault_label=f_label, cavity_conf=math.nan,
                              fault_conf=math.nan, label_source="test", start=100, n_samples=(1000 * 5))
        with self.assertRaises(RuntimeError):
            wex.load_data()

    def test_factory(self):
        # Test out no defaults
        f = Factory()
        ex = f.get_example(e_type=ExampleType.EXAMPLE, zone="1L24",
                           dt=datetime.datetime.strptime("2000_01_01 000001.1", "%Y_%m_%d %H%M%S.%f"),
                           cavity_label=None, fault_label=None, cavity_conf=None, fault_conf=None, label_source="test")

        wex = f.get_example(e_type=ExampleType.WINDOWED_EXAMPLE, zone="1L24",
                            dt=datetime.datetime.strptime("2000_01_01 000001.1", "%Y_%m_%d %H%M%S.%f"),
                            cavity_label=None, fault_label=None, cavity_conf=None, fault_conf=None, label_source="test",
                            start=-105, n_samples=500)

        self.assertEqual(type(ex).__name__, "Example")
        self.assertEqual(type(wex).__name__, "WindowedExample")

        # Test out with defaults
        f = Factory(e_type=ExampleType.EXAMPLE)
        ex = f.get_example(zone="1L24", dt=datetime.datetime.strptime("2000_01_01 000001.1", "%Y_%m_%d %H%M%S.%f"),
                           cavity_label=None, fault_label=None, cavity_conf=None, fault_conf=None, label_source="test")

        f = Factory(e_type=ExampleType.WINDOWED_EXAMPLE, start=-105, n_samples=500)
        wex = f.get_example(zone="1L24", dt=datetime.datetime.strptime("2000_01_01 000001.1", "%Y_%m_%d %H%M%S.%f"),
                            cavity_label=None, fault_label=None, cavity_conf=None, fault_conf=None, label_source="test")

        self.assertEqual(type(ex).__name__, "Example")
        self.assertEqual(type(wex).__name__, "WindowedExample")

        # Check that not supplying an e_type raises a ValueError
        with self.assertRaises(ValueError):
            f = Factory()
            f.get_example()

    def check_data_on_disk(self, e):
        """Check that the event_df capture files were save to disk"""
        self.assertTrue(os.path.exists(e.get_event_path()))  # Does event directory exist?
        self.assertEqual(8, len(os.listdir(e.get_event_path())))  # Were the capture files all made?

    def check_event_df(self, e):
        """Check that the event_df of Example e looks OK"""
        self.assertIsNotNone(e.event_df)  # Did we get anything at all
        self.assertTupleEqual(e.event_df.shape,
                              (8192, 1 + 17 * 8))  # 8192 time samples, 1 time column + 17 signals/cavity


if __name__ == '__main__':
    unittest.main()
