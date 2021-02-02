import math
import tarfile

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


class Example:
    """A class representing a (SME) labeled fault event.

    zone - CED string (e.g., 1L21)
    timestamp - timestamp string of format ts_fmt
    cavity_label - an int label specifying the cavity that caused fault
    fault_label - the string label specifying the type of fault that occurred
    ts_fmt - strftime format specifier
    data_dir - where can the root of the data directory be found on the local filesystem
    """

    # A regex for matching
    capture_file_regex = re.compile(r"R.*harv\..*\.txt")

    def __init__(self, zone, dt, cavity_label, fault_label, cavity_conf, fault_conf, label_source, data_dir=None):
        """Expects timestamp to be YYYY-mm-dd HH:MM:SS.f formatted string and CED zone (1L21)

        Note: The data_dir defaults to the local data/waveforms/... path.  data/waveforms is a symlink which can either
        removed to save in place or updated to point to another location on the local system.  I use it to point to my
        local SSD while the project lives in my NFS mounted home directory.
        """

        # Expert/model provided labels
        self.cavity_label = cavity_label
        self.fault_label = fault_label
        self.cavity_conf = cavity_conf
        self.fault_conf = fault_conf
        self.label_source = label_source

        # Zone and timestamp info for the example event
        self.event_datetime = dt
        self.event_zone = zone

        # Will eventually hold the waveform data from the event
        self.event_df = None
        self.data_dir = data_dir

    def get_event_path(self):
        data_dir = self.data_dir if self.data_dir is not None else Config().data_dir
        return os.path.join(data_dir, self.event_zone, self.get_file_system_time_string())

    def get_event_path_compressed(self):
        """Return the expected path the event if it has been compressed (i.e. tar/gzipped)."""
        return f"{self.get_event_path()}.tar.gz"

    def load_data(self, verbose=False):
        """A catch-all for all activity needed to finish constructing the Example data"""
        if verbose:
            print("loading data - " + str(self))
        self.retrieve_event_df()

        # Some early events had a bug where the Time column was wrong.  Flip the order and change sign to fix.
        if self.event_df.Time[0] > -1000.0:
            if Config().debug:
                print(f"{self.event_zone} {self.event_datetime}: Found flipped Time column.  Fixing.")
            self.event_df.Time = -1 * self.event_df.Time.values[::-1]

    def unload_data(self, verbose=False):
        """A catch-all for deleting the Examples data (event_df) from memory"""
        if verbose:
            print("unloading data - " + str(self))
        self.event_df = None

    def get_file_system_time_string(self):
        """Return the file system formatted time string.

        Well almost, because Tom doesn't record the fractional seconds so we have to figure that part out on the fly.
        """
        return self.event_datetime.strftime("%Y_%m_%d/%H%M%S.%f")[:-5]

    def get_normal_time_string(self):
        """Return a 'normally' formatted time string (not quite ISO-8601) formatted time string.

        Down to 0.1 since that is as much as we record on the file system. The fractional part is probably wrong since
        Tom does not save that in the label files.
        """
        return self.event_datetime.strftime("%Y-%m-%d_%H:%M:%S.%f")[:-5]

    def get_web_time_strings(self, fmt="%Y-%m-%d %H:%M:%S"):
        """Return the web api formatted time strings (begin, end).  Needs to be formatted for url queries.

        Since Tom only has down to the second, we have to query the API for begin and end times that will surround the
        event.
        """

        begin = self.event_datetime
        end = self.event_datetime + datetime.timedelta(seconds=1)
        return begin.strftime(fmt), end.strftime(fmt)

    def retrieve_event_df(self):
        """Get the event waveform data and save it into event_df.  Saves capture files to disk after retrieval.

        First tries to get it from disk if it exists.  If not, then it downloads it from the web and saves it to disk.
        Only drawback from downloading it from the web is that is loses the per-capture file timestamps, and it is
        slower.

        Note: first clears any existing data, i.e., self.event_df.
        """

        self.event_df = None
        # Try to get the data from the web if we don't already have it
        if self.capture_files_on_disk():
            # Load up the data into event_df
            self.event_df = self.retrieve_event_df_from_disk()
        elif self.compressed_capture_files_on_disk():
            # Uncompress and load the data into event_df
            self.event_df = self.retrieve_event_df_from_disk(compressed=True)
        else:
            self.event_df = self.download_waveforms_from_web()

    def download_waveforms_if_needed(self):
        """Downloads the fault waveform data and save it if it is not already present on disk."""
        if not self.capture_files_on_disk():
            self.download_waveforms_from_web()

    def download_waveforms_from_web(self, data_server='accweb.acc.jlab.org'):
        """Downloads the data from accweb for the specified zone and timestamp.

        This has to do some guesstimating about which event to download because of imprecise time stamps.  Also access
        to accweb requires that you be on a JLab network (VPN should be fine, but probably not the guest wifi).

        Returns a viable event_df waveform DataFrame
        """

        # Setup to download the data
        base = 'https://' + data_server + '/wfbrowser/ajax/event?'
        z = urllib.parse.quote_plus(self.event_zone)
        (begin, end) = self.get_web_time_strings()
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

    def retrieve_event_df_from_disk(self, compressed=False):
        """Loads the cached copy of event's capture file into a DataFrame and returns it."""
        return Example.parse_event_dir(event_path=self.get_event_path(), compressed=compressed)

    @staticmethod
    def is_capture_file(filename):
        """Validates if filename appears to be a valid capture file.

            Args:
                filename (str): The name of the file that is to be validated

            Returns:
                bool: True if the filename appears to be a valid capture file.  Otherwise False.
        """
        return Example.capture_file_regex.match(filename)

    @staticmethod
    def parse_capture_file(file):
        """Parses an individual capture file into a Pandas DataFrame object.

        Reads all data in as float64 dtypes because a column of all integers will default to integers (e.g., all zeroes)

            Args:
                file (file): A file like object.  Either the string of the filename or a file_like_object

            Returns:
                DataFrame: A pandas DataFrame containing the data from the specified capture file
        """
        return pd.read_csv(file, sep="\t", comment='#', skip_blank_lines=True,
                           dtype='float64')

    def set_event_df(self):
        """"""

    @staticmethod
    def parse_event_dir(event_path, compressed=False):
        """Parses the  capture files in the BaseModel's event_dir and sets event_df to the appropriate pandas DataFrame.

        The waveform names are converted from <EPICS_NAME><Waveform> (e.g., R123WFSGMES), to <Cavity_Number>_<Waveform>
        (e.g., 3_GMES).  This allows analysis code to more easily handle waveforms from different zones.

            Returns:
                None

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
                    zone_df = zone_df.join(Example.parse_capture_file(os.path.join(event_path, filename)).set_index("Time"),
                                           on="Time")

        # Now format the column names to remove the zone information but keep a cavity and signal identifiers
        zone_df.columns = Example.convert_waveform_column_names(zone_df.columns)

        return zone_df

    @staticmethod
    def convert_waveform_column_names(columns):
        """Turns waveform PV names (R1M1WFSGMES) into more uniform name (1_GMES)

        Expects a list of waveform columns from within a single zone, i.e., a list of event waveform names to convert
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

    def save_event_df_to_disk(self, event_df):
        """This method is saves the event waveform DataFrame to disk.  Can provide faster access to 'raw' data later.

        If capture files already exist, it won't try to overwrite them.  Does nothing if event_path is None.
        """

        # Create the event directory tree
        if not os.path.exists(self.get_event_path()):
            os.makedirs(self.get_event_path())

        # Get the capture file name components
        date = self.event_datetime.strftime("%Y_%m_%d")
        time = self.event_datetime.strftime("%H%M%S.%f")
        base = event_df.columns.values[3][:3]

        # Make all of the capture files for cavities in the downloaded data
        for i in range(1, 9):
            cav = base + str(i)
            cav_columns = ['Time'] + [col for col in event_df.columns.values if cav in col]
            out_file = os.path.join(self.get_event_path(), "{}WFSharv.{}_{}.txt".format(cav, date, time))
            if not os.path.exists(out_file):
                event_df[cav_columns].to_csv(out_file, index=False, sep='\t')

    def remove_event_df_from_disk(self):
        """Deletes the 'cached' event waveform data for this event from disk."""

        # Remove the event directory if uncompressed
        if os.path.exists(self.get_event_path()):
            shutil.rmtree(self.get_event_path())

        # Remove the tar.gz compressed event directory if on disk
        if self.compressed_capture_files_on_disk():
            os.unlink(self.get_event_path_compressed())

        return

    def capture_files_on_disk(self):
        """Checks if captures files are currently saved to disk"""
        return os.path.exists(self.get_event_path()) and len(os.listdir(self.get_event_path())) > 0

    def compressed_capture_files_on_disk(self):
        """Checks if captures files are currently saved to disk in a compressed format"""
        return os.path.exists(self.get_event_path_compressed())

    def has_matching_labels(self, example):
        """Check if the supplied example has the same cavity and fault type label"""
        if example is not None:
            if self.fault_label == example.fault_label and self.cavity_label == example.cavity_label:
                return True
        return False

    def plot_waveforms(self, signals=None, downsample=32):
        """Plot the waveform data associated with this example.

        Args:
            signals (list(str)) - A list of signal names to plot, e.g. '1_GMES'.  If None, then GMES, DETA2, GASK, CRFP,
                                  and PMES will be plotted for all cavities
            downsample (integer) - The downsampling factor, i.e., keep every <downsample>-th point.  By default keep
                                   every 16th point
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
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

            i += 1

        # Add the main title and display
        fig.suptitle(f"{self.event_zone} {self.event_datetime} - cav={self.cavity_label}, fault={self.fault_label} ({self.label_source})")
        plt.show()

    def to_string(self):
        """This provides a more descriptive string than __str__"""
        return f"<zone:{self.event_zone}  ts:{self.event_datetime}  cav_label:{self.cavity_label}  fault_label:" \
               f"{self.fault_label}  cav_conf:{self.cavity_conf}  fault_conf:{self.fault_conf}  " \
               f"label_source:{self.label_source}>"

    def __eq__(self, other):
        """Determines equality by zone, datetime, labels, and confidence values."""
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

    def __key(self):
        return self.event_zone, self.event_datetime, self.cavity_label, self.cavity_conf, self.fault_label, \
               self.fault_conf

    def __hash__(self):
        """Returns the hash of """
        return hash(self.__key())

    def __ne__(self, other):
        """Determines inequality.  Inverse of __eq__"""
        return not self == other

    def __str__(self):
        return f"<{self.event_zone} {self.event_datetime}>"
