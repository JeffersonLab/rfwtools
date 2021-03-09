import unittest
from rfwtools import utils
import json


class Utils(unittest.TestCase):
    def test_get_events_from_web(self):
        exp = '{"events": [{"id": 119239, "datetime_utc": "2020-03-01 23:15:36.3", "location": "2L24", "system": ' \
              '"rf", "archive": false, "classification": "", "captureFiles": [], "labels": null}]}'
        result = json.dumps(utils.get_events_from_web(begin="2020-03-01 18:15:00", end="2020-03-01 18:16:00"))
        self.assertEqual(exp, result)

    def test_get_signal_names(self):
        exp = ['1_GMES', '1_PMES', '2_GMES', '2_PMES']
        result = utils.get_signal_names(['1', '2'], ['GMES', 'PMES'])
        self.assertListEqual(exp, result)


if __name__ == '__main__':
    unittest.main()
