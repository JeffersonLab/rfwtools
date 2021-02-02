import math
import unittest
import os
from datetime import datetime
from unittest import TestCase
import pandas as pd

from rfwtools.example_validator import ExampleValidator
from rfwtools.example_set import ExampleSet
from rfwtools.example import Example
import test
from rfwtools.config import Config

# Prime the pump on the timestamp map.
test.load_timestamp_map()

# Update the config object to reflect these paths
Config().label_dir = test.test_label_dir
Config().output_dir = test.test_output_dir
Config().data_dir = os.path.join(test.test_data_dir, 'tmp')


class TestValidator(ExampleValidator):
    #A very simple validator for testing purposes.

    def validate_data(self, deployment="ops"):
        """It's not valid if it's not zone 1L24"""
        if self.event_zone != '1L24':
            raise ValueError("Only 1L24 faults are valid during testing!")


class TestExampleSet(TestCase):
    old_config = None

    @classmethod
    def setUpClass(cls):
        cls.old_config = Config().dump_yaml_string()

    @classmethod
    def tearDownClass(cls):
        Config().load_yaml_string(cls.old_config)

    def test_update_example_set(self):
        es = ExampleSet()
        es.add_label_file_data(label_files=('test1.txt', 'test2.txt'))
        es.remove_duplicates_and_mismatches()

        # Empty df should fail column names check
        self.assertRaises(ValueError, es.update_example_set, df=pd.DataFrame())
        # Column names should pass, but dtypes check should fail
        self.assertRaises(ValueError, es.update_example_set, df=pd.DataFrame(
            {"zone": [], "dtime": [], "cavity_label": [], "cavity_conf": [], "fault_label": [], "fault_conf": [],
             "label_source": []}))

        df = es.get_example_df()
        df.loc[1, "fault_conf"] = 0.73
        df.loc[1, "cavity_conf"] = 0.52

        # Make sure these don't equal
        self.assertFalse(df.equals(es.get_example_df()))

        # Update with the new data
        es.update_example_set(df=df)

        # Now they should match
        self.assertTrue(df.equals(es.get_example_df()))
        self.assertEqual(len(df), 3)

    def test_process_label_files_with_excludes(self):
        excludes = ['0L04', '1L23']
        es = ExampleSet()
        es.add_label_file_data(label_files=('test1.txt', 'test2.txt'), exclude_zones=excludes)
        for z in es.get_example_df()["zone"]:
            for zone in excludes:
                self.assertFalse(z == zone, "exclude: {}  event: {}\n".format(zone, z))

    def test_add_label_file_data(self):
        es = ExampleSet()
        es.add_label_file_data(label_files=('test1.txt', 'test2.txt'))

        self.assertEqual(5, es.count_events())
        self.assertEqual(10, es.count_labels())
        self.assertEqual(3, es.count_duplicated_events())
        self.assertEqual(8, es.count_duplicated_labels())
        self.assertEqual(6, es.count_mismatched_labels())
        self.assertEqual(2, es.count_duplicated_events_with_mismatched_labels())

        exp_report = """#### Summary ####
Note: event == unique zone/timestamp, label == row in label_file

Number of events: 5
Number of labels: 10
Number of events with multiple labels: 3
Number of duplicate labels: 8
Number of 'extra' labels: 5

Number of events with mismatched labels: 2
Number of mismatched labels: 6

#### Events With Mismatched Labels ####
   zone                   dtime cavity_label   fault_label  cavity_conf  fault_conf                            example label_source
3  1L25 2020-01-08 09:13:07.800            5  Microphonics          NaN         NaN  <1L25 2020-01-08 09:13:07.800000>    test1.txt
4  1L25 2020-01-08 09:13:07.800            5  Microphonics          NaN         NaN  <1L25 2020-01-08 09:13:07.800000>    test1.txt
5  1L25 2020-01-08 09:13:07.800            5  Microphonics          NaN         NaN  <1L25 2020-01-08 09:13:07.800000>    test1.txt
6  1L24 2020-01-08 11:24:21.300            7  Quench_100ms          NaN         NaN  <1L24 2020-01-08 11:24:21.300000>    test1.txt
7  1L24 2020-01-08 11:24:21.300            8  Quench_100ms          NaN         NaN  <1L24 2020-01-08 11:24:21.300000>    test1.txt
1  1L25 2020-01-08 09:13:07.800            6  Microphonics          NaN         NaN  <1L25 2020-01-08 09:13:07.800000>    test2.txt 
"""
        self.assertEqual(exp_report, es.get_label_file_report())

    def test_remove_duplicate_and_mismatches(self):
        es = ExampleSet()
        es.add_label_file_data(label_files=('test1.txt', 'test2.txt'))

        # Make sure we have duplicates, etc. at this point
        self.assertTrue(es.count_duplicated_events() > 0, "Checking that duplicates DO exist")

        # Now remove those duplicates
        es.remove_duplicates_and_mismatches()

        # Make sure we get what we expect
        ts_fmt = '%Y-%m-%d %H%M%S.%f'
        # 1L23	0	25	Multi Cav turn off	2020/01/08 09:12:53
        # 1L24	1	33	Single Cav Turn off	2020/01/08 09:13:00
        # 1L24	7	39	Quench_100ms	2020/01/08 11:24:32
        exp = (
            Example(zone='1L23', dt=datetime.strptime("2020-01-08 091253.3", ts_fmt),
                    cavity_label='0', fault_label='Multi Cav turn off',
                    cavity_conf=math.nan, fault_conf=math.nan, label_source="test1.txt"),
            Example(zone='1L24', dt=datetime.strptime("2020-01-08 091300.6", ts_fmt),
                    cavity_label='1', fault_label='Single Cav Turn off',
                    cavity_conf=math.nan, fault_conf=math.nan, label_source="test1.txt"),
            Example(zone='1L24', dt=datetime.strptime("2020-01-08 112432.5", ts_fmt),
                    cavity_label='7', fault_label='Quench_100ms', cavity_conf=math.nan,
                    fault_conf=math.nan, label_source="test2.txt")
        )

        # Check that the examples match
        for ex in es.get_example_df()["example"]:
            self.assertTrue(ex in exp)
        for ex in exp:
            self.assertTrue(ex in es.get_example_df()["example"].values)

    def test_purge_invalid_examples(self):
        es = ExampleSet()

        # This is using the larger label file
        es.add_label_file_data(label_files=('test3.txt',))
        es.remove_duplicates_and_mismatches()
        es.purge_invalid_examples(TestValidator(), download_data=False, progress=False, report=False)

        # Verify that the only values in there are for zone 1L24
        exp = ('1L24',)
        self.assertTupleEqual(exp, tuple(es.get_example_df()['zone'].unique()))

    def test_add_web_service_data(self):
        # Test that this doesn't blow up and that we get the expected number of examples

        es = ExampleSet()

        # This should produce 14 examples at the very end of the March 2020 run.
        es.add_web_service_data(begin=datetime.strptime("2020-03-23", "%Y-%m-%d"),
                                end=datetime.strptime("2020-03-24", "%Y-%m-%d"))

        self.assertEqual(14, len(es.get_example_df()))


if __name__ == '__main__':
    unittest.main()
