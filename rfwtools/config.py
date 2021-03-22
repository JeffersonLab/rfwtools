"""This module is used for managing the configuration of rfwtools.

This module contains as single class, a Singleton, that manages the configuration of many parts of rfwtools.  This
includes the definition of several resource locations, debug behavior, and filters on valid data.

Basic Usage:
::

    from rfwtools.config import Config
    config = Config()
    config.output_dir = '/path/to/my/save/files/'

    # Or equivalently

    Config().output_dir = '/path/to/my/save/files/'

Config File Example:

Most values are straightforward.  exclude_times is a list of lists where null implies None.
::

    app_dir: /projects/rfw-stuff
    data_dir: /projects/rfw-stuff/data/waveforms/data/rf
    data_server: accweb.acc.jlab.org
    debug: true
    exclude_times:
    - - 2020-01-01 12:34:56.700000
      - 2020-01-02 12:34:56.700000
    - - 2020-01-03 12:34:56.700000
      - null
    exclude_zones:
    - 1L07
    label_dir: /projects/rfw-stuff/data/labels
    output_dir: /projects/rfw-stuff/processed-output
    wfb_base_url: wfbrowser
"""
from datetime import datetime
import os
from typing import Any, List, Tuple

import yaml


class Config:
    """A singleton class for containing application configuration.  Written as a singleton to enable easy extension.

    Attributes:
        instance:
            The internal configuration object.
        app_dir:
            A string defining the root directory of the application.  Currently used in initial setup only.  Defaults
            to the current working directory.
        data_dir:
            A string defining the path to waveform data.  Defaults to CWD/data/waveforms/data/rf.
        debug:
            A boolean that controls debug output.  Defaults to False (meaning no debug output).
        label_dir:
            A string defining the directory that holds label files. Defaults to CWD/data/labels
        output_dir:
            A string defining the directory to read/write saved files produced by this package.  Defaults to
            CWD/processed-output.
        exclude_zones:
            A list of strings that define zones to exclude from parsing operations.  Defaults to ['0L04']
        exclude_times:
            A list of 2-tuples of datetime.datetime objects.  Each tuple is a range of time to exclude from analysis.
            Both endpoints are inclusive, and a value of  None is interpreted as +/- Inf.  Defaults to None.
        data_server:
            A string for the server hostname to contact for the web-based data API.  Defaults to accweb.acc.jlab.org.
        wfb_base_url:
            A string that is the base URL (context root) of the waveform browser web app.  Defaults to wfbrowser.
    """

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
            if os.path.exists('./rfwtools.cfg'):
                Config.read_config_file()

    @staticmethod
    def dump_yaml_string() -> str:
        """Write config out to a YAML formatted string.

        Note: the nested Class causes trouble with pickle so this is a reasonable alternative in most scenarioes.

        Returns:
            The YAML-formatted configuration string.
        """
        return yaml.dump(Config.instance.__dict__)

    @staticmethod
    def load_yaml_string(string: str) -> None:
        """Read in a YAML formatted string containing configuration information.

        This method overwrites only the values defined in the string.

        Arguments:
            string: The YAML-formatted string to parse.
        """
        cfg = yaml.safe_load(string)
        for key in cfg.keys():
            if key == 'exclude_times':
                try:
                    Config.__validate_exclude_times(cfg[key])
                except Exception as exc:
                    print(f"Exception processing exclude_times.  Setting it to None.\n{exc}")
                    Config.instance.__dict__[key] = None
            Config.instance.__dict__[key] = cfg[key]

    @staticmethod
    def __validate_exclude_times(exclude_times: List[Tuple[datetime, datetime]]) -> None:
        """Validates the structure and types of exclude_times.

        Arguments:
            exclude_times: A list of tuples of datetime objects or None.

        Raises:
            ValueError: if
        """
        if exclude_times is None:
            return None
        e_times = list()
        if type(exclude_times).__name__ != 'list' and type(exclude_times).__name__ != 'tuple':
            raise ValueError("Received unexpected exclude_times format.")

        for values in exclude_times:
            if len(values) != 2:
                raise ValueError("Range does not have two values")
            if type(values).__name__ != 'list':
                raise ValueError("Range is not a list.")

            start, end = values
            if start is not None and type(start).__name__ != 'datetime':
                raise ValueError("start should be of type datetime or None")
            if end is not None and type(end).__name__ != 'datetime':
                raise ValueError("end should be of type datetime or None")

    @staticmethod
    def write_config_file(file: str) -> None:
        """Writes out the current configuration to the specified file.

        Arguments:
            file: The name file the name to write configuration infomration to.
        """

        # Make sure the singleton config exists
        Config()
        with open(file, mode="w") as f:
            f.write(Config.dump_yaml_string())

    @staticmethod
    def read_config_file(file: str = None) -> None:
        """Parses a YAML-formatted config file and updates internal configuration.

        Relative files will be considered relative to the current working directory.

        Arguments:
            file:
                Path to the file to read.  Relative paths are considered as relative to the current working directory.
                Default value is rfwtools.cfg
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

    def __getattr__(self, name: str) -> Any:
        """Redirect unresolved attribute queries to the single instance."""
        return getattr(Config.instance, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Redirect attribute modification to the single instance."""
        return setattr(Config.instance, name, value)
