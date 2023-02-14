import unittest
import pandas as pd
import os
from datetime import datetime

from rfwtools import utils
from rfwtools.example import Example
from rfwtools.extractor import down_sample

zone = '1L22'
ts = "02/24/2019, 04:22:01"
ts_fmt = "%m/%d/%Y, %H:%M:%S"
cav_label = 4
f_label = "Heat_Riser_Choke"
test_dir = os.path.dirname(os.path.realpath(__file__))
tmp_data_dir = os.path.join(test_dir, "..", "test-data", "tmp")
test_data_dir = os.path.join(test_dir, "..", "test-data")


class MyDownSample(unittest.TestCase):
    def test_down_sample_extractor(self):
        e = Example(zone=zone, dt=datetime.strptime(ts, ts_fmt), cavity_label=cav_label, fault_label=f_label,
                    cavity_conf=None, fault_conf=None, label_source="test", data_dir=tmp_data_dir)
        e.load_data()

        signals = utils.get_signal_names(cavities=['1', '2', '3', '4', '5', '6', '7', '8'], waveforms=['GMES', 'PMES'])
        result = down_sample.down_sample_extractor(e, signals=signals, step_size=4096)
        e.unload_data()

        expected = pd.DataFrame([0.25326273, 0.25230953, 0.2141799, 0.21371006, 0.18991698, 0.15107737,
                                 -0.1109384, -0.15199239, 0.26061878, 0.25737597, -0.12775754, -0.12775754,
                                 0.25775753, 0.25634663, 0.2400413, 0.2400413, 0.24496047, 0.24029828,
                                 0.14094824, 0.14130977, 0.23896194, 0.24245187, 0.22675633, 0.2273138,
                                 0.26968984, 0.27072888, -0.20100573, -0.20100573, 0.25899339, 0.25971385,
                                 0.15478684, 0.15561521]).T.round(5)

        self.assertTrue(expected.equals(result.round(5)))

    def test_lttb_extractor(self):
        e = Example(zone=zone, dt=datetime.strptime(ts, ts_fmt), cavity_label=cav_label, fault_label=f_label,
                    cavity_conf=None, fault_conf=None, label_source="test", data_dir=tmp_data_dir)
        e.load_data()

        signals = utils.get_signal_names(cavities=['1', '2', '3', '4'], waveforms=['GMES', 'PMES'])
        result = down_sample.lttb_extractor(e, signals=signals, n_out=4)
        e.unload_data()

        expected = pd.DataFrame(
            [17.1984, 17.2091, 17.1993, 12.5585, 167.333, 167.333, 174.419, -33.0853, 17.3064, 17.2774, 16.5212,
             17.3892, -18.9459, -19.0338, -1.43921, -19.1766, 20.2104, 20.2104, 20.2082, 13.8623, -64.9017, -64.9567,
             -179.808, 9.77783, 15.9987, 16.0055, 18.0546, 0., 161.252, 161.263, 177.358, -140.405]).T.round(5)

        self.assertTrue(expected.equals(result.round(5)))

        if __name__ == '__main__':
            unittest.main()
