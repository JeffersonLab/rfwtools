import math
import os

from unittest import TestCase

from rfwtools.example import Example, IExample
from rfwtools.extractor.windowing import window_extractor
import tests
from rfwtools.config import Config
import pandas as pd
import datetime

# Prime the pump on the timestamp map.
tests.load_timestamp_map()

# Update the config object to reflect these paths
Config().label_dir = tests.test_label_dir
Config().output_dir = tests.test_output_dir
Config().data_dir = tests.tmp_data_dir


class TestExample(Example):
    """A dummy test Example object that is easy to understand and has a small amount of data"""

    def __init__(self):
        super().__init__(zone='asdf',
                         dt=datetime.datetime(year=2020, month=3, day=1, hour=0, minute=0, second=0, microsecond=0),
                         cavity_label='adsf', fault_label='asdf', cavity_conf=1.0, fault_conf=1.0, label_source='asdf')

        # Setup the results of the various extraction routines.  All start assume window_start = 0 and size = 3.

        # No standardizing, no down sampling
        self.win_0_3_df = pd.DataFrame(
            {'window_label': pd.Categorical(['test']), 'window_start': [0], 'n_samples': [3], 'window_min': [0.1],
             'window_max': [0.3], 'window_standardized': [False], 'window_downsampled': [False],
             'window_downsample_size': [None],
             'Sample_1_1_sig': [1], 'Sample_2_1_sig': [2], 'Sample_3_1_sig': [3],
             'Sample_1_2_sig': [2], 'Sample_2_2_sig': [5], 'Sample_3_2_sig': [6]})

        # Standardized, no down sample
        self.win_0_3_standard_df = pd.DataFrame(
            {'window_label': pd.Categorical(['test']), 'window_start': [0], 'n_samples': [3], 'window_min': [0.1],
             'window_max': [0.3], 'window_standardized': [True], 'window_downsampled': [False],
             'window_downsample_size': [None],
             'Sample_1_1_sig': [-1.22474487], 'Sample_2_1_sig': [0.],
             'Sample_3_1_sig': [1.22474487], 'Sample_1_2_sig': [-1.37281295],
             'Sample_2_2_sig': [0.39223227], 'Sample_3_2_sig': [0.98058068]})

        # No standard scaling, down sampled to two points
        self.win_0_3_ds_2_df = pd.DataFrame(
            {'window_label': pd.Categorical(['test']), 'window_start': [0], 'n_samples': [3], 'window_min': [0.1],
             'window_max': [0.3], 'window_standardized': [False], 'window_downsampled': [True],
             'window_downsample_size': [2],
             'Sample_1_1_sig': [1.], 'Sample_2_1_sig': [3.], 'Sample_1_2_sig': [2.],
             'Sample_2_2_sig': [6.66666667]})

        # Standardized, down sampled to two points
        self.win_0_3_standard_ds_2_df = pd.DataFrame(
            {'window_label': pd.Categorical(['test']), 'window_start': [0], 'n_samples': [3], 'window_min': [0.1],
             'window_max': [0.3], 'window_standardized': [True], 'window_downsampled': [True],
             'window_downsample_size': [2],
             'Sample_1_1_sig': [-1.22474487], 'Sample_2_1_sig': [1.22474487],
             'Sample_1_2_sig': [-1.37281295], 'Sample_2_2_sig': [1.37281295]})

    def load_data(self):
        """Setup a simple event_df."""
        self.event_df = pd.DataFrame({'Time': [0.1, 0.2, 0.3, 0.4, 0.5],
                                      '1_sig': [1, 2, 3, 4, 5],
                                      '2_sig': [2, 5, 6, 7, 9]})


class TestWindowExtractor(TestCase):

    def test_window_extractor_simple(self):
        # Test against a simplified version of an example
        sigs = ['1_sig', '2_sig']
        ex = TestExample()

        # Test a simple window
        exp = ex.win_0_3_df
        res = window_extractor(ex, windows={'test': 0}, n_samples=3, signals=sigs, standardize=False,
                               downsample=False)
        pd.testing.assert_frame_equal(exp, res)

        # Test a standardized window
        exp = ex.win_0_3_standard_df
        res = window_extractor(ex, windows={'test': 0}, n_samples=3, signals=sigs, standardize=True,
                               downsample=False)
        pd.testing.assert_frame_equal(exp, res)

        # Test a down sampled window
        exp = ex.win_0_3_ds_2_df
        res = window_extractor(ex, windows={'test': 0}, n_samples=3, signals=sigs, standardize=False,
                               downsample=True, ds_kwargs={'num': 2, 'axis': 0})
        pd.testing.assert_frame_equal(exp, res)

        # Test a down sampled window
        exp = ex.win_0_3_standard_ds_2_df
        res = window_extractor(ex, windows={'test': 0}, n_samples=3, signals=sigs, standardize=True,
                               downsample=True, ds_kwargs={'num': 2, 'axis': 0})
        pd.testing.assert_frame_equal(exp, res)

    def test_window_extractor_real(self):
        # Test that we get expected feature extraction results on a small set of data.

        # Read in the expected values.  There are lots of features here.
        exp = pd.read_csv(os.path.join(tests.test_data_dir, "test-window-feature_set.csv"), comment="#", index_col=False)
        exp.window_label = pd.Categorical(exp.window_label, categories=['good', 'bad'])

        # Calculate the result
        fmt = "%Y-%m-%d %H:%M:%S.%f"
        dt1 = datetime.datetime.strptime("2020-01-08 09:12:53.3", fmt)
        dt2 = datetime.datetime.strptime("2020-01-08 09:13:00.6", fmt)
        ex1 = Example(zone="1L23", dt=dt1, cavity_label="0", fault_label="Multi Cav turn off", cavity_conf=math.nan,
                      fault_conf=math.nan, label_source="test")
        ex2 = Example(zone="1L24", dt=dt2, cavity_label="1", fault_label="Single Cav Turn off", cavity_conf=math.nan,
                      fault_conf=math.nan, label_source="test")
        examples = (ex1, ex2)
        windows = {'good': -1536, 'bad': -105}
        result = pd.concat([window_extractor(ex, windows=windows, n_samples=500) for ex in examples], axis=0,
                           ignore_index=True)

        #result.to_csv(os.path.join(tests.test_data_dir, "test-window-feature_set.csv"), index=False)

        # Uses pandas testing routine to see if they are equal
        pd.testing.assert_frame_equal(exp.drop(columns=['window_downsample_size']),
                                      result.drop(columns=['window_downsample_size']))
