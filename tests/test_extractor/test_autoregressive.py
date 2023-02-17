import math
import os

from unittest import TestCase

from rfwtools.example import Example
from rfwtools.extractor.autoregressive import autoregressive_extractor
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


class TestARExtractor(TestCase):

    def test_autoregressive_extractor(self):
        # Test that we get expected feature extraction results on a small set of data.

        # Read in the expected values.  There are lots of features here.
        exp = pd.read_csv(os.path.join(tests.test_data_dir, "test-AR-feature_set.csv"), comment="#")

        # Calculate the result
        fmt = "%Y-%m-%d %H:%M:%S.%f"
        dt1 = datetime.datetime.strptime("2020-01-08 09:12:53.3", fmt)
        dt2 = datetime.datetime.strptime("2020-01-08 09:13:00.6", fmt)
        ex1 = Example(zone="1L23", dt=dt1, cavity_label="0", fault_label="Multi Cav turn off", cavity_conf=math.nan,
                      fault_conf=math.nan, label_source="test")
        ex2 = Example(zone="1L24", dt=dt2, cavity_label="1", fault_label="Single Cav Turn off", cavity_conf=math.nan,
                      fault_conf=math.nan, label_source="test")
        examples = (ex1, ex2)
        result = pd.concat([autoregressive_extractor(ex) for ex in examples], axis=0, ignore_index=True)

        # Uses pandas testing routine to see if they are equal
        pd.testing.assert_frame_equal(exp, result)
