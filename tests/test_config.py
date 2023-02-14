from datetime import datetime

from unittest import TestCase
from rfwtools.config import Config
import os

# Location where test data can be saved.
test_out = os.path.join(Config().app_dir, "test", 'test-output')


class TestConfig(TestCase):
    old_config = None

    @classmethod
    def setUpClass(cls):
        cls.old_config = Config().dump_yaml_string()

    @classmethod
    def tearDownClass(cls):
        Config().load_yaml_string(cls.old_config)

    def test_basic_use(self):
        # Check that changes to one are seen across all (i.e., Singleton works)
        conf1 = Config()
        conf2 = Config()

        conf1.debug = 15   # Normally a bool
        conf1.output_dir = 'asdf'

        self.assertEqual(conf1.debug, conf2.debug)
        self.assertEqual(conf1.output_dir, conf2.output_dir)

    def test_export_import_yaml(self):
        # Testing that we can effectively (de)serialize a config object
        filename = os.path.join(test_out, "config-test.pkl")

        # Make a Config object and change some defaults
        conf = Config()
        conf.debug = True
        conf.exclude_zones = ['1L07']
        fmt = "%Y-%m-%d %H:%M:%S.%f"
        t1 = datetime.strptime("2020-01-01 12:34:56.7", fmt)
        t2 = datetime.strptime("2020-01-02 12:34:56.7", fmt)
        t3 = datetime.strptime("2020-01-03 12:34:56.7", fmt)
        t4 = None
        times = [[t1, t2], [t3, t4]]
        conf.exclude_times = times

        # Export config to yaml string
        string = conf.dump_yaml_string()

        # Change the values again to detect if load works
        conf.debug = False
        conf.exclude_times = None
        conf.exclude_zones = ["2L22"]

        # Import config from yaml string
        conf.load_yaml_string(string)

        self.assertEqual(conf.debug, True)
        self.assertListEqual(conf.exclude_zones, ["1L07"])
        self.assertListEqual(conf.exclude_times, times)
