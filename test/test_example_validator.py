from unittest import TestCase
from rfwtools.example_validator import ExampleValidator, WindowedExampleValidator
from rfwtools.example import Example, WindowedExample
from datetime import datetime

import os


def check_list_equal(list1, list2):
    return len(list1) == len(list2) and sorted(list1) == sorted(list2)


ts_fmt = "%Y_%m_%d %H%M%S.%f"
data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test-data')


class TestExampleValidator(TestCase):
    # Example of a simplified event.  Allows for easier checking that the data parsing is correct
    simple_event_example = Example(zone="1L24", dt=datetime.strptime("2000_01_01 000001.1", ts_fmt), cavity_label=None,
                                   fault_label=None, cavity_conf=None, fault_conf=None, label_source="test",
                                   data_dir=os.path.join(data_dir, "short-test"))

    # Example that has the proper event path starting at the date directory.  One with good control modes, one with bad
    good_mode_example = Example(zone="1L25", dt=datetime.strptime("2018_10_04 052657.4", ts_fmt), cavity_label=None,
                                fault_label=None, cavity_conf=None, fault_conf=None, label_source="test",
                                data_dir=os.path.join(data_dir, "good-cavity-mode"))
    bad_mode_example = Example(zone="1L25", dt=datetime.strptime("2018_10_04 052659.4", ts_fmt), cavity_label=None,
                               fault_label=None, cavity_conf=None, fault_conf=None, label_source="test",
                               data_dir=os.path.join(data_dir, "bad-cavity-mode"))

    # Examples of the normal data file we will be working with
    good_example = Example(zone="1L24", dt=datetime.strptime("2020_01_08 091300.6", ts_fmt), cavity_label=None,
                           fault_label=None, cavity_conf=None, fault_conf=None, label_source="test",
                           data_dir=os.path.join(data_dir, "good-example"))
    good_meta_example = Example(zone="1L25", dt=datetime.strptime("2018_10_05 044555.3", ts_fmt), cavity_label=None,
                                fault_label=None, cavity_conf=None, fault_conf=None, label_source="test",
                                data_dir=os.path.join(data_dir, "good-example-meta"))

    # Contain proper capture files, but have problems which waveforms are present
    missing_example = Example(zone="1L25", dt=datetime.strptime("2018_10_05 044555.3", ts_fmt), cavity_label=None,
                              fault_label=None, cavity_conf=None, fault_conf=None, label_source="test",
                              data_dir=os.path.join(data_dir, "missing-waveforms"))
    duplicate_example = Example(zone="1L25", dt=datetime.strptime("2018_10_05 044555.3", ts_fmt), cavity_label=None,
                                fault_label=None, cavity_conf=None, fault_conf=None, label_source="test",
                                data_dir=os.path.join(data_dir, "duplicate-waveforms"))

    # Has missing or duplicate capture files
    missing_cfs_example = Example(zone="1L25", dt=datetime.strptime("2018_10_05 044408.2", ts_fmt), cavity_label=None,
                                  fault_label=None, cavity_conf=None, fault_conf=None, label_source="test",
                                  data_dir=os.path.join(data_dir, "missing-cfs"))
    missing_cfs_example2 = Example(zone="1L25", dt=datetime.strptime("2018_10_05 044555.3", ts_fmt), cavity_label=None,
                                   fault_label=None, cavity_conf=None, fault_conf=None, label_source="test",
                                   data_dir=os.path.join(data_dir, "missing-cfs"))
    missing_cfs_example3 = Example(zone="2L26", dt=datetime.strptime("2020_01_14 073632.7", ts_fmt), cavity_label=None,
                                   fault_label=None, cavity_conf=None, fault_conf=None, label_source="test",
                                   data_dir=os.path.join(data_dir, "missing-cfs"))
    duplicate_cfs_example = Example(zone="1L25", dt=datetime.strptime("2018_10_05 044408.2", ts_fmt), cavity_label=None,
                                    fault_label=None, cavity_conf=None, fault_conf=None, label_source="test",
                                    data_dir=os.path.join(data_dir, "duplicate-cfs"))

    # Has a mismatched Time column between R1P1 and R1P2
    mismatched_time_example = Example(zone="1L25", dt=datetime.strptime("2018_10_05 044555.3", ts_fmt),
                                      cavity_label=None, fault_label=None, cavity_conf=None, fault_conf=None,
                                      label_source="test", data_dir=os.path.join(data_dir, "mismatched-times"))
    bad_time_interval_example = Example(zone="1L25", dt=datetime.strptime("2018_10_05 044555.3", ts_fmt),
                                        cavity_label=None, fault_label=None, cavity_conf=None, fault_conf=None,
                                        label_source="test", data_dir=os.path.join(data_dir, "bad-time-interval"))

    def test_validate_capture_file_counts(self):
        # Should work
        ev = ExampleValidator()
        ev.set_example(TestExampleValidator.good_example)
        ev.validate_capture_file_counts()

        # Test out failure cases
        ev.set_example(TestExampleValidator.missing_cfs_example)
        self.assertRaises(ValueError, ev.validate_capture_file_counts)

        ev.set_example(TestExampleValidator.missing_cfs_example2)
        self.assertRaises(ValueError, ev.validate_capture_file_counts)

        ev.set_example(TestExampleValidator.missing_cfs_example3)
        self.assertRaises(ValueError, ev.validate_capture_file_counts)

        ev.set_example(TestExampleValidator.duplicate_cfs_example)
        self.assertRaises(ValueError, ev.validate_capture_file_counts)

    def test_validate_capture_file_waveforms(self):
        # This should work and not raise an exception
        ev = ExampleValidator()
        ev.set_example(TestExampleValidator.good_example)
        ev.validate_capture_file_waveforms()

        # This should work and not raise an exception
        ev.set_example(TestExampleValidator.good_meta_example)
        ev.validate_capture_file_waveforms()

        # Check that the validation raises an exception on files missing waveforms
        ev.set_example(TestExampleValidator.missing_example)
        self.assertRaises(ValueError, ev.validate_capture_file_waveforms)

        # Check that the validation raises an exception on files with duplicate waveforms
        ev.set_example(TestExampleValidator.duplicate_example)
        self.assertRaises(ValueError, ev.validate_capture_file_waveforms)

    def test_validate_waveform_times(self):
        # This should work
        ev = ExampleValidator()
        ev.set_example(TestExampleValidator.good_example)
        ev.validate_waveform_times(step_size=0.2, delta_max=0.012)

        # This should work
        ev.set_example(TestExampleValidator.good_meta_example)
        ev.validate_waveform_times(step_size=0.05, delta_max=0.0011)

        # This should raise a ValueError since two of the files have different time columns
        ev = ExampleValidator()
        ev.set_example(TestExampleValidator.mismatched_time_example)
        self.assertRaises(ValueError, ev.validate_waveform_times)

        # This should raise a ValueError since all of the files have the same Time series that is too long
        ev.set_example(TestExampleValidator.bad_time_interval_example)
        self.assertRaises(ValueError, ev.validate_waveform_times)

    def test_validate_cavity_modes(self):
        # The time associated with this event should have all good control modes
        ev = ExampleValidator()
        ev.set_example(TestExampleValidator.good_mode_example)
        ev.validate_cavity_modes()

        # The time for this should have at least one bad control mode and will raise and exception
        ev.set_example(TestExampleValidator.bad_mode_example)
        self.assertRaises(ValueError, ev.validate_cavity_modes)

    def test_validate_data(self):
        # Test overall validation of Examples
        ev = ExampleValidator()
        ev.set_example(TestExampleValidator.good_example)
        ev.validate_data()

        ex = WindowedExample(zone="1L24", dt=datetime.strptime("2020_01_08 091300.6", ts_fmt), cavity_label=None,
                             fault_label=None, cavity_conf=None, fault_conf=None, label_source="test",
                             data_dir=os.path.join(data_dir, "good-example"), start=-200, n_samples=250*5)
        wev = WindowedExampleValidator()
        wev.set_example(ex)
        wev.validate_data()
