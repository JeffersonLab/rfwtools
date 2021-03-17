"""A package for managing the data associated with a single fault event.

The has a single class that manages basic data access for an individual fault event Example.  If the data is not found
in the specified location, then it will attempt to download the data and create the appropriate directory structure.
This functionality typically requires access to JLab's internal networks (e.g., VPN).

Expected data structure is <data_dir>/<zone>/<date>/<timestamp>/<capture files>.  Alternatively, event data may be
compressed at the <timestamp> directory level, i.e. <timestamp>.tar.gz.

Typical usage example:

::

   import Example
   e = Example(zone="1L23", dt=datetime.strptime("%Y-%m-%d %H:%M:%S.%f"), cavity_label="1", fault_label="Microphonics",
               cavity_conf=math.nan, fault_conf=math.nan, label_source='my_label_file.txt')
   e.load_data()
   df = e.event_df
   e.unload_data()
   # Now do something with the data

"""

import math
import tarfile
from enum import Enum
from typing import Tuple, Dict, List

import requests
import os
import datetime
import re
import shutil
import urllib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from rfwtools.network import SSLContextAdapter
from rfwtools.config import Config


class ExampleType(Enum):
    """The types of supported IExample types"""
    EXAMPLE = 1
    WINDOWED_EXAMPLE = 2


class IExample:
    """Abstract class that defines the basic interface for an Example.

    This only defines methods and properties related to data access.
    """

    def __init__(self, data_dir=None):
        """Construct an instance of the Example class."""

        # Will eventually hold the waveform data from the event
        self.event_df = None
        self.data_dir = data_dir
        self.e_type = None

    def load_data(self) -> None:
        """This method should do everything necessary to create an example_df DataFrame at a minimum.

        Subclasses should use this method as the interface for loading all data associated with the fault event.
        """
        raise NotImplementedError("This method has not been implement by child class")

    def unload_data(self) -> None:
        """This method should do everything necessary to free up memory used in load_data."""
        raise NotImplementedError("This method has not been implement by child class")

    def get_example_type(self) -> ExampleType:
        """Get this Example's ExampleType.

        Return:
            The Enum corresponding to the class type
        """
        return self.e_type


class Factory:
    """A class for producing a variety of types of IExample objects"""

    def __init__(self, e_type: ExampleType = None, **kwargs):
        """Construct a factory for creating Example objects with a default type.

        All additional keyword arguments will be passed to the Example type constructor.

        Arguments:
            e_type:
                The default type of IExample object that will be created.
            **kwargs:
                All additional keyword arguments will be passed to the responsible constructor on every get_example()
                call.
        """
        self.e_type = e_type
        self.example_kwargs = kwargs

    def get_example(self, e_type: ExampleType = None, **kwargs) -> IExample:
        """Get an IExample object.

        Arguments:
            e_type:
                The type of IExample object to create.  If None, defaults to the e_type given at construction.
            **kwargs:
                All additional keyword arguments will be passed to the responsible constructor on every get_example()
                call.

        Raises:
              ValueError: When no e_type is available or if an unsupported e_type is specified.
        """
        t = None
        if e_type is not None:
            t = e_type
        else:
            t = self.e_type

        if t is None:
            raise ValueError("No e_type specified at construction or get_example.")

        if t == ExampleType.EXAMPLE:
            ex = Example(**kwargs, **self.example_kwargs)
        elif t == ExampleType.WINDOWED_EXAMPLE:
            ex = WindowedExample(**kwargs, **self.example_kwargs)
        else:
            raise ValueError(f"Unsupported ExampleType {e_type}")

        return ex


class Example(IExample):
    """A class representing a (SME) labeled fault event.  Manages data access and can download missing data.

    Harvester fault data typically occupies ~10 MB of memory/disk space, and collections of these events often number in
    the thousands.  Holding all of this event data in memory is typically not an option.  Additionally, the data is
    found in multiple files organized in a directory tree, saved as a single tar.gz file, or is downloadable from web
    services.  This class provides methods for easing access of this data and quickly loading/unloading it from memory.

    If the data is not found in the specified location, then it will attempt to download the data and create the
    appropriate directory structure.  This functionality typically requires access to JLab's internal networks (e.g.,
    VPN).

    Expected data structure is <data_dir>/<zone>/<date>/<timestamp>/<capture files>.  Alternatively, event data may be
    compressed at the <timestamp> directory level, i.e. <timestamp>.tar.gz.

    Attributes:

    zone: A string identifying the zone in CED format (e.g., 1L21)
    dt: datetime object matching the local time of the fault event
    cavity_label: a string label specifying the cavity that caused fault (typically "0", "1", ..., "8")
    fault_label: a string label specifying the type of fault that occurred (ExampleSet has a list of "known" labels")
    cavity_conf: A floating point number in [0, 1] representing the probability/confidence placed in the cavity label
    fault_conf: A floating point number in [0, 1] representing the probability/confidence placed in the fault label
    label_source: The source of the labels.  Typically either label files or the output of a model
    data_dir: string defining filesystem path under which data can be found.  If None, Config().data_dir is used.
    """

    # A regex for matching
    capture_file_regex = re.compile(r"R.*harv\..*\.txt")

    def __init__(self, zone:str , dt: datetime, cavity_label: str, fault_label: str, cavity_conf: float,
                 fault_conf: float, label_source: str, data_dir:str =None):
        """Construct an instance of the Example class."""

        super().__init__(data_dir=data_dir)

        # Expert/model provided labels
        self.cavity_label = cavity_label
        self.fault_label = fault_label
        self.cavity_conf = cavity_conf
        self.fault_conf = fault_conf
        self.label_source = label_source

        # Zone and timestamp info for the example event
        self.event_datetime = dt
        self.event_zone = zone

        #: The type of example this is.
        self.e_type = ExampleType.EXAMPLE


    def load_data(self, verbose: bool = False) -> None:
        """Top-level method for loading data associated with Example instance.

        Some early waveforms were saved with the Time column essentially inverted.  This method checks for and fixes
        that problem by means a simple criteria.  If Time[0] > -1000, then the Time array is flipped.

        Arguments:
            verbose: Should extra information be printed to STDOUT
        """
        if verbose:
            print("loading data - " + str(self))
        self._retrieve_event_df()

        # Some early events had a bug where the Time column was wrong.  Flip the order and change sign to fix.
        if self.event_df.Time[0] > -1000.0:
            if Config().debug:
                print(f"{self.event_zone} {self.event_datetime}: Found flipped Time column.  Fixing.")
            self.event_df.Time = -1 * self.event_df.Time.values[::-1]

    def unload_data(self, verbose: bool = False) -> None:
        """Top-level method for deleting the Examples data (event_df) from memory.

        Arguments:
            verbose: Should extra information be printed to STDOUT
        """
        if verbose:
            print("unloading data - " + str(self))
        self.event_df = None

    def get_event_path(self, compressed: bool = False) -> str:
        """Generates the expected location for uncompressed event waveform data.

        Arguments:
            compressed: Should the returned path be for a compressed (tgz) event

        Returns:
            The expected path to uncompressed directory of waveform data.
        """
        data_dir = self.data_dir if self.data_dir is not None else Config().data_dir

        path = os.path.join(data_dir, self.event_zone, self._get_file_system_time_string())
        if compressed:
            return f"{path}.tar.gz"
        else:
            return path

    def get_capture_file_contents(self) -> Dict[str, str]:
        """Creates a dictionary of capture file content.  Typically eight files, each is ~1 MB.

        Returns:
            A dictionary of capture file content keyed on the name of the capture file.
        """

        content = {}
        # Directly read each file into the dictionary
        if self.capture_files_on_disk(compressed=False):
            for filename in os.listdir(self.get_event_path(compressed=False)):
                if Example.is_capture_file(filename):
                    with open(os.path.join(self.get_event_path(compressed=False), filename), "r") as f:
                        content[filename] = f.read()

        # Wend our way through the tarfile and read the content
        elif self.capture_files_on_disk(compressed=True):
            with tarfile.open(self.get_event_path(compressed=True), mode="r:gz") as f:
                for member in f.getmembers():
                    if member.isfile():
                        # Make sure to remove the base directory from the name
                        file = os.path.basename(member.name)
                        if Example.is_capture_file(file):
                            content[file] = f.extractfile(member)

        return content

    def _get_file_system_time_string(self) -> str:
        """Return the file system formatted time string.

        A common issue here is that Tom records faults down to the second, while the filesystem records them to the
        tenth of a second.  This method simply returns the Example's timestamp (dt) as a string in the proper format.


        Note :The ExamlpeSet class has functionality for determining the proper timestamp from the the waveform browser
        service.

        Returns:
            The timestamp formatted in as expected in the accelerator filesystem. e.g. "2020_01_28/234556.1".
        """
        return self.event_datetime.strftime("%Y_%m_%d/%H%M%S.%f")[:-5]

    def _get_normal_time_string(self) -> str:
        """Return a 'normally' formatted time string (not quite ISO-8601) formatted time string.

        Accuracy down to 0.1 since that is as much as we record on the file system.

        Returns:
            The event timestamp in local time and nearly ISO-8601 - "%Y-%m-%d_%H:%M:%S.%f".
        """
        return self.event_datetime.strftime("%Y-%m-%d_%H:%M:%S.%f")[:-5]

    def _get_web_time_strings(self, fmt: str = "%Y-%m-%d %H:%M:%S") -> Tuple[str, str]:
        """Generates two datetime strings that bound the fault at one second accuracy.

        Many label files do not record the timestamps down to the tenth of a second as is needed by the web interface.
        This method generates the two timestamps that should bound the actual event time.  These are not URL formatted.

        Arguments:
            fmt: datetime.strftime format string
        Returns:
            Tuple of the web API formatted time strings (begin, end).  Needs to be encoded for url queries.

        Since Tom only has down to the second, we have to query the API for begin and end times that will surround the
        event.
        """

        begin = self.event_datetime
        end = self.event_datetime + datetime.timedelta(seconds=1)
        return begin.strftime(fmt), end.strftime(fmt)

    def _retrieve_event_df(self) -> None:
        """Get the event waveform data and save it into event_df.  Saves capture files to disk after retrieval.

        First tries to get it from disk if it exists.  If not, then it downloads it from the web and saves it to disk.
        Only drawback from downloading it from the web is that is loses the per-capture file timestamps, and it is
        slower.

        Note: first clears any existing data, i.e., self.event_df.
        """

        self.event_df = None
        # Try to get the data from the web if we don't already have it
        if self.capture_files_on_disk(compressed=False):
            # Load up the data into event_df
            self.event_df = Example.parse_event_dir(event_path=self.get_event_path(compressed=False), compressed=False)
        elif self.capture_files_on_disk(compressed=True):
            # Uncompress and load the data into event_df
            self.event_df = Example.parse_event_dir(event_path=self.get_event_path(compressed=True), compressed=True)
        else:
            self.event_df = self._download_waveforms_from_web()

    def _download_waveforms_from_web(self, data_server: str = None, wfb_base_url: str = None) -> pd.DataFrame:
        """Downloads the data from accweb for the specified zone and timestamp.

        This has to do some guesstimating about which event to download because of imprecise time stamps.  Also access
        to accweb requires that you be on a JLab network (VPN should be fine, but probably not the guest wifi).

        Arguments:
            data_server:
                The hostname of the server to query for the event data.  If None, Config.data_server is used.
            wfb_base_url:
                The base URL/context root of the web-based data app.  If None, Config.wfb_base_rul is used.  E.g.,
                "wfbrowser"

        Returns:
             A viable event_df waveform DataFrame
        """

        # Use default config values if none supplied.
        if data_server is None:
            data_server = Config().data_server
        if wfb_base_url is None:
            wfb_base_url = Config().wfb_base_url

        # Setup to download the data
        base = f'https://{data_server}/{wfb_base_url}/ajax/event?'
        z = urllib.parse.quote_plus(self.event_zone)
        (begin, end) = self._get_web_time_strings()
        b = urllib.parse.quote_plus(begin)
        e = urllib.parse.quote_plus(end)
        url = base + 'out=csv&includeData=true&location=' + z + '&begin=' + b + '&end=' + e

        # Download the data - supply the session/SSLContextAdapter to use Windows trust store
        s = requests.Session()
        adapter = SSLContextAdapter()
        s.mount(url, adapter)
        r = s.get(url)

        # Test if we got a good status code.
        if not r.status_code == 200:
            raise RuntimeError("Received non-ok response - " + r.status_code)
        if r.content == "":
            raise RuntimeError("Received empty content from  - ")

        # Read the data in from the response stream.  The web api gives you one big CSV file for the whole zone
        data = StringIO(r.text.replace('time_offset', 'Time'))
        event_df = pd.read_csv(data)

        # Should save the event with the original PVs to keep data on disk looking as usual
        self.save_event_df_to_disk(event_df)

        # Convert the column names to be the standard generic names
        event_df.columns = Example.convert_waveform_column_names(event_df.columns)

        return event_df

    @staticmethod
    def is_capture_file(filename: str) -> bool:
        """Validates if filename appears to be a valid capture file.

            Args:
                filename (str): The name of the file that is to be validated

            Returns:
                bool: True if the filename appears to be a valid capture file.  Otherwise False.
        """
        return Example.capture_file_regex.match(filename)

    @staticmethod
    def parse_capture_file(file: str) -> pd.DataFrame:
        """Parses an individual capture file into a Pandas DataFrame object.

        Reads all data in as float64 dtypes because a column of all integers will default to integers (e.g., all zeroes)

            Args:
                file (file): A file like object.  Either the string of the filename or a file_like_object

            Returns:
                DataFrame: A pandas DataFrame containing the data from the specified capture file
        """
        return pd.read_csv(file, sep="\t", comment='#', skip_blank_lines=True, dtype='float64')

    @staticmethod
    def parse_event_dir(event_path: str, compressed: bool = False) -> None:
        """Parses the  capture files in the BaseModel's event_dir and sets event_df to the appropriate pandas DataFrame.

        The waveform names are converted from <EPICS_NAME><Waveform> (e.g., R123WFSGMES), to <Cavity_Number>_<Waveform>
        (e.g., 3_GMES).  This allows analysis code to more easily handle waveforms from different zones.

        Arguments:
            event_path: The path to the event directory or compressed tar.gz file
            compressed: Is the data a compressed tar.gz file or a regular directory
        Raises:
             ValueError: if a column name is discovered with an unexpected format
        """
        zone_df = None

        if compressed:
            # Here we open the tarfile in memory, only opening the members whose name matches the capture file pattern
            with tarfile.open(event_path) as tar:
                for member in reversed(tar.getmembers()):
                    if not Example.is_capture_file(os.path.basename(member.name)):
                        continue
                    if zone_df is None:
                        zone_df = Example.parse_capture_file(tar.extractfile(member))
                    else:
                        zone_df = zone_df.join(Example.parse_capture_file(tar.extractfile(member)).set_index('Time'),
                                               on="Time")
        else:
            for filename in sorted(os.listdir(event_path)):
                # Only try to process files that look like capture files
                if not Example.is_capture_file(filename):
                    continue
                if zone_df is None:
                    zone_df = Example.parse_capture_file(os.path.join(event_path, filename))
                else:
                    # Join the existing zone data with the new capture file by using the "Time" column as an index to
                    # match rows
                    zone_df = zone_df.join(
                        Example.parse_capture_file(os.path.join(event_path, filename)).set_index("Time"), on="Time")

        # Now format the column names to remove the zone information but keep a cavity and signal identifiers
        zone_df.columns = Example.convert_waveform_column_names(zone_df.columns)

        return zone_df

    @staticmethod
    def convert_waveform_column_names(columns: List[str]) -> List[str]:
        """Turns waveform PV names (R1M1WFSGMES) into more uniform name based on cavity and waveform (1_GMES)

        Arguments:
            columns: List of waveform columns from a single zone, i.e., a list of event waveform names to convert.

        Returns:
            The updated/standardized column names sans zone identifier.
        """
        pattern = re.compile(r'R\d\w\dWF[TS]')
        new_columns = []
        for column in columns:
            if column != "Time":
                # This only works for PV/waveform names of the proper format.  That's all we should be working with.
                if not pattern.match(column):
                    raise ValueError("Found unexpected waveform data - " + column)
                column = column[3] + "_" + column[7:]
            new_columns.append(column)
        return new_columns

    def save_event_df_to_disk(self, event_df: pd.DataFrame) -> None:
        """This method is saves the event waveform DataFrame to disk.  Can provide faster access to 'raw' data later.

        If capture files already exist, it won't try to overwrite them.  Does nothing if event_path is None.  Note that
        every capture file will end up with the same timestamp as self.event_datetime.

        Arguments:
            event_df: The DataFrame for which we should create a fault event directory of capture files.
        """

        # Do nothing if compressed file exists
        if os.path.exists(self.get_event_path(compressed=True)):
            return

        # Create the event directory tree
        if not os.path.exists(self.get_event_path(compressed=False)):
            os.makedirs(self.get_event_path(compressed=False))

        # Get the capture file name components
        date = self.event_datetime.strftime("%Y_%m_%d")
        time = self.event_datetime.strftime("%H%M%S.%f")
        base = event_df.columns.values[3][:3]

        # Make all of the capture files for cavities in the downloaded data
        for i in range(1, 9):
            cav = base + str(i)
            cav_columns = ['Time'] + [col for col in event_df.columns.values if cav in col]
            out_file = os.path.join(self.get_event_path(compressed=False),
                                    "{}WFSharv.{}_{}.txt".format(cav, date, time))
            if not os.path.exists(out_file):
                event_df[cav_columns].to_csv(out_file, index=False, sep='\t')

    def remove_event_df_from_disk(self) -> None:
        """Deletes the 'cached' event waveform data for this event from disk.  Both compressed and uncompressed data."""

        # Remove the event directory if uncompressed
        if self.capture_files_on_disk(compressed=False):
            shutil.rmtree(self.get_event_path(compressed=False))

        # Remove the tar.gz compressed event directory if on disk
        if self.capture_files_on_disk(compressed=True):
            os.unlink(self.get_event_path(compressed=True))

        return

    def capture_files_on_disk(self, compressed: bool = False) -> bool:
        """Checks if captures files are currently saved to disk.

        Arguments:
            compressed: Are we checking for compressed file (True), or uncompressed (False, default)?

        Returns:
            True if the compressed file or regular directors were found.
        """
        if compressed:
            return os.path.exists(self.get_event_path(compressed=True))
        else:
            return os.path.exists(self.get_event_path(compressed=False)) and \
                   len(os.listdir(self.get_event_path(compressed=False))) > 0

    def has_matching_labels(self, example: 'Example') -> bool:
        """Check if the supplied example has the same cavity and fault type label.

        Arguments:
            example: A Example object to compare labels against

        Returns:
            True if both cavity and fault labels match.  False otherwise.
        """
        if example is not None:
            if self.fault_label == example.fault_label and self.cavity_label == example.cavity_label:
                return True
        return False

    def plot_waveforms(self, signals: List[str] = None, downsample: int = 32) -> None:
        """Plot the waveform data associated with this example.  Optionally down sample the signals.

        Args:
            signals:
                A list of signal names to plot, e.g. '1_GMES'.  If None, then GMES, DETA2, GASK, CRFP, and PMES will be
                plotted for all cavities
            downsample:
                The down sampling factor, i.e., keep every <downsample>-th point.  By default keep every 32nd point
        """

        # Make sure we've downloaded the data if needed and loaded it into memory
        if self.event_df is None:
            self.load_data()

        # Create the default set of signals.  This matching approach is preferable to an explicit list in case some
        # some cavities are missing
        if signals is None:
            signals = []
            for wf in ('GMES', 'DETA2', 'GASK', 'CRFP', 'PMES'):
                for col in self.event_df.columns:
                    if col.endswith(wf):
                        signals.append(col)

        # Get the unique set of waveforms (GMES, CRRP, etc.) each one will get it's own plot
        waveforms = set(wf[2:] for wf in signals)

        # Create a data structure that holds the signals needed for each waveform plot
        plot_signals = {}
        for wf in waveforms:
            plot_signals[wf] = [signal for signal in signals if signal.endswith(wf)]

        # Let make a single multi-plot that has square-ish dimensions.  This is the size we're shooting for
        ncols = math.ceil(math.sqrt(len(plot_signals)))
        nrows = math.ceil(len(plot_signals) / ncols)

        fig, axn = plt.subplots(nrows, ncols, sharex="all", figsize=(8 * ncols, 3 * nrows))
        i = 1

        # Set the lines to be a little narrower
        sns.set(rc={'lines.linewidth': 0.7})

        for wf in sorted(plot_signals):
            # Select the axis to draw on
            ax = plt.subplot(nrows, ncols, i)

            plot_df = self.event_df[['Time'] + plot_signals[wf]]

            # SNS likes the long DF format over the wide DF format.  Use pd.melt to convert, value/variable are the
            # default column names after melting
            sns.lineplot(x='Time', y='value', hue='variable', data=pd.melt(plot_df.iloc[::downsample, :], ['Time']))
            ax.set_title(wf + f" - down sampled {downsample}:1")
            plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

            i += 1

        # Add the main title and display
        fig.suptitle(f"{self.event_zone} {self.event_datetime} - cav={self.cavity_label}, fault={self.fault_label} "
                     f"({self.label_source})")
        plt.subplots_adjust(left=0.05, right=0.925, wspace=0.35, hspace=0.35)
        plt.show()

    def to_string(self) -> str:
        """This provides a more descriptive string than __str__.

        Returns:
            A string representation of the example including zone, time, label info, and label source."""
        return f"<zone:{self.event_zone}  ts:{self.event_datetime}  cav_label:{self.cavity_label}  fault_label:" \
               f"{self.fault_label}  cav_conf:{self.cavity_conf}  fault_conf:{self.fault_conf}  " \
               f"label_source:{self.label_source}>"

    def __eq__(self, other: 'Example') -> bool:
        """Determines equality by zone, datetime, labels, and confidence values.

        Arguments:
            other: An Example object to compare.

        Returns:
            True if the Examples are equivalent, False otherwise.
        """
        if other is not None:
            if self.event_datetime != other.event_datetime:
                return False
            if self.event_zone != other.event_zone:
                return False
            if self.cavity_label != other.cavity_label:
                return False
            if self.fault_label != other.fault_label:
                return False
            if not Example.__float_equal(self.cavity_conf, other.cavity_conf):
                return False
            if not Example.__float_equal(self.fault_conf, other.fault_conf):
                return False

        return True

    @staticmethod
    def __float_equal(x, y):
        """A smarter equality check for floating point numbers.  This considers nan == nan as True"""
        if x is None:
            return y is None
        if y is None:
            return x is None

        return (x == y) or (math.isnan(x) and math.isnan(y))

    def __key(self) -> Tuple[str, datetime.datetime, str, float, str, float]:
        """Returns essential attributes of the Example."""
        return self.event_zone, self.event_datetime, self.cavity_label, self.cavity_conf, self.fault_label, \
               self.fault_conf

    def __hash__(self) -> int:
        """Returns the hash of self's key attributes.  Needed for inclusion in dictionaries.

        Returns:
            The results of hash.
        """
        return hash(self.__key())

    def __ne__(self, other: 'Example') -> bool:
        """Determines inequality.  Inverse of __eq__"""
        return not self == other

    def __str__(self) -> str:
        """Returns a short string representing this Example - only zone and datetime."""
        return f"<{self.event_zone} {self.event_datetime}>"


class WindowedExample(Example):
    """An extension of Example class that allows the caller to specify only a time-window of event_df be returned.

    This window is based on the relative values in the Time column.  The standard time range is approximately
    [-1500, 100], but this is variable depending on control system settings.  An Exception is raised at load time if the
    specified time range is not a strict subset of the Examples Time column.
    """

    def __init__(self, zone: str, dt: datetime, cavity_label: str, fault_label: str, cavity_conf: float,
                 fault_conf: float, label_source: str, start: float, n_samples: int, data_dir: str = None):
        """Construct an instance th will only store the required window upon a load_data() call.

        Arguments:
            start: The start of the time window.
            n_samples: The number of samples to include after the start of the window.
        """
        super().__init__(zone, dt, cavity_label, fault_label, cavity_conf, fault_conf, label_source, data_dir)
        self.e_type = ExampleType.WINDOWED_EXAMPLE

        #: The start of the window relative to the fault onset
        self.start = start

        #: The number of samples requested after the start value
        self.n_samples = n_samples

        #: The last Time value in the window.  Determined after loading data for the first time.
        self.end = None

    def load_data(self, verbose: bool = False) -> None:
        """Load the fault event data according to Example.load_data() and retain only the defined time window.

        This changes the Time column from being relative to the fault onset to being relative to the start of the
        window.  This means that the first time value is unlikely to be exactly 0, but should be a small positive
        number.

        Arguments:
            verbose: Should extra information be printed during operation

        Raises:
            RuntimeError: If the specified window of data is not available.
        """
        super().load_data(verbose)

        # Check that start is in the data
        if self.event_df.Time.min() > self.start:
            raise RuntimeError("Requested window is before start of event_df data.")
        if self.event_df.Time.max() < self.start:
            raise RuntimeError("Requested window is after end of event_df data.")

        # Compute some importance index values
        start_i = self.event_df[self.event_df["Time"] >= self.start].Time.idxmin()
        end_i = start_i + self.n_samples
        i_min = self.event_df.index.min()
        i_max = self.event_df.index.max()
        t_min = self.event_df.Time[i_min]
        t_max = self.event_df.Time[i_max]

        # Check if the requested window is in the data
        if not (self.start >= t_min and end_i <= i_max):
            raise RuntimeError(f"Requested window of {self.start} plus {self.n_samples} samples, but event data is"
                               f"[{t_min}, {t_max}]")

        # Grab the window
        self.event_df = self.event_df.iloc[start_i:end_i, :]

        # Set the end time before changing the Time column meaning
        self.end = self.event_df.Time.max()

        # Convert the Time column to be relative to the start of the window, not the fault onset.
        self.event_df.Time = self.event_df.Time - self.start
