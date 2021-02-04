import os
import yaml


class Config:
    """A singleton class for containing application configuration.  Written as a singleton to enable easy extension"""

    class __Config:
        """This private inner class is used to implement the singleton interface"""

        def __init__(self):
            # The base directory of the application.
            self.app_dir = os.path.realpath(os.getcwd())

            # The path to the root of the data directory for all events (similar to /usr/opsdata/waveforms/data/rf)
            self.data_dir = os.path.join(self.app_dir, "data", "waveforms", "data", "rf")

            # Controls how much information is printed during processing
            self.debug = False

            # Directory containing label files
            self.label_dir = os.path.join(self.app_dir, 'data', 'labels')

            # Directory to use for saving file output
            self.output_dir = os.path.join(self.app_dir, "processed-output")

            # Default zones to exclude from sources
            self.exclude_zones = ["0L04"]

            # Default time ranges to exclude from sources
            self.exclude_times = None

            # Default hostname of the production waveform browser web server
            self.data_server = 'accweb.acc.jlab.org'

            # Default URL for the waveform browser (wfbrowser)
            self.wfb_base_url = "wfbrowser"

        def __str__(self):
            return Config.dump_yaml_string()

    instance = None

    def __init__(self):
        """Only make an instance of the inner Config object if its missing"""
        if Config.instance is None:
            Config.instance = Config.__Config()

    @staticmethod
    def dump_yaml_string():
        """Write config out to a YAML formatted string.  Workaround - the nested Class causes trouble with pickle."""
        return yaml.dump(Config.instance.__dict__)

    @staticmethod
    def load_yaml_string(string):
        Config.instance.__dict__ = yaml.safe_load(string)

    @staticmethod
    def write_config_file(file):
        """Writes out the current configuration to the specified file."""

        # Make sure the singleton config exists
        Config()
        with open(file, mode="w") as f:
            f.write(Config.dump_yaml_string())

    @staticmethod
    def read_config_file(file=None):
        """Parses a YAML-formatted config file.  Defaults to rfwtools.cfg in the current working directory.

        Relative files will be considered relative to the current working directory.
        """

        if file is None:
            path = os.path.join(os.getcwd(), 'rfwtools.cfg')
        elif os.path.isabs(file):
            path = file
        else:
            path = os.path.join(os.getcwd(), file)

        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found - '{path}'")

        with open(path, mode="r") as f:
            Config.load_yaml_string(f.read())

    def __getattr__(self, name):
        """Redirect unresolved attribute queries to the single instance"""
        return getattr(Config.instance, name)

    def __setattr__(self, name, value):
        """Redirect attribute modification to the single instance"""
        return setattr(Config.instance, name, value)
