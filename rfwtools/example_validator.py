import datetime
import itertools
import os
import re

from rfwtools import mya
from rfwtools.example import Example


class ExampleValidator:
    """This class provides functionality for checking that an individual example meets the criteria for validity.

    Note: This class assumes the data is present on disk to check at the location specified by the supplied example.

    Some checks are very basic, e.g., do we have all of the necessary data from the fault event.  Others are a bit more
    nuanced, e.g., was the cavity in the proper RF mode.
    """

    def __init__(self):
        """Create an instance for validating Example."""
        self.event_path = None
        self.event_datetime = None
        self.event_zone = None
        self.event_df = None

    def set_example(self, example):
        """Set internal information about the example to validate."""
        self.event_path = example.get_event_path()
        self.event_datetime = example.event_datetime
        self.event_zone = example.event_zone

        # Need to load the data and make an internal copy for later use.
        example.load_data()
        self.event_df = example.event_df.copy()
        example.unload_data()

    def validate_data(self, deployment='ops'):
        """Check that the event directory and it's data is of the expected format.

        This method inspects the event directory and raises an exception if a problem is found.  The following aspects
        of the event directory and waveform data are validated.
           # All eight cavities are represented by exactly one capture file
           # All of the required waveforms are represented exactly once
           # All of the capture files use the same timespan and have constant sampling intervals
           # All of the cavity are in the appropriate control mode (GDR I/Q => 4)

        Returns:
            None: Subroutines raise an exception if an error condition is found.

        """
        self.validate_capture_file_counts()
        self.validate_capture_file_waveforms()
        self.validate_waveform_times()
        self.validate_cavity_modes(deployment=deployment)
        self.validate_zones()

    def validate_capture_file_counts(self):
        """This method checks that we have exactly one capture file per cavity/IOC.

        The harvester grouping logic coupled with unreliable IOC behavior seems to produce fault event directories where
        either an IOC has multiple capture files or are missing.  We want to make sure we have exactly eight capture
        files - one per IOC.  Raises an exception in the case that something is amiss.

            Returns:
                None

            Raises:
                ValueError: if either missing or "duplicate" capture files are found.
        """

        # Count capture files per cavity
        capture_file_counts = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0}
        for filename in os.listdir(self.event_path):
            cavity = filename[3]
            if cavity not in capture_file_counts.keys():
                raise ValueError("Found capture file for an unsupported cavity - " + cavity)

            capture_file_counts[cavity] += 1

        for cavity in capture_file_counts.keys():
            if capture_file_counts[cavity] > 1:
                raise ValueError("Duplicate capture files exist for zone '" + cavity + "'")
            if capture_file_counts[cavity] == 0:
                raise ValueError("Missing capture file for zone '" + cavity + "'")

    def validate_capture_file_waveforms(self):
        """Checks that all of the required waveforms are present exactly one time across all capture files.

        If event_df is None, then the capture files themselves are loaded.  If event_df is not None, then the files are
        checked directly.

            Returns:
                None

            Raises:
                ValueError: if any required waveform is repeated
        """

        # Get a structure for counting matches of waveforms
        req_signals = ["IMES", "QMES", "GMES", "PMES", "IASK", "QASK", "GASK", "PASK", "CRFP", "CRFPP",
                       "CRRP", "CRRPP", "GLDE", "PLDE", "DETA2", "CFQE2", "DFQES"]

        # Since we have the event_df from the event, first compare the DataFrame columns to the required signals
        # If we don't have the event_df for some reason, try to look at the files on disk
        if self.event_df is not None:
            # Generate the list of required and actual columns
            req_columns = [i + j for i, j in itertools.product(("1_", "2_", "3_", "4_", "5_", "6_", "7_", "8_"),
                                                               req_signals)]
            req_columns.insert(0, "Time")
            columns = [i for i in self.event_df.columns.values]  # Don't want to modify the actual event_df

            # Sort them so we can do element wise comparison
            req_columns.sort()
            columns.sort()

            if len(req_columns) != len(columns) or req_columns != columns:
                raise ValueError("Found event_df does not have the required waveform columns.")

        else:
            # Will contain regex's that are used to check for required waveforms, and the count of how many matches
            req_waveforms = {re.compile("Time"): 0}
            for sig in req_signals:
                for cav in [1, 2, 3, 4, 5, 6, 7, 8]:
                    wf = r"R\d\w" + str(cav) + "WF[ST]" + sig + "$"
                    req_waveforms[re.compile(wf)] = 0

            # Metadata are lines at the top of the file that start with a #.  Probably no spaces, but just to be safe
            metadata_regex = re.compile(r"^\s*#")

            # Go through each capture file and make sure that the required waveforms are present
            for filename in os.listdir(self.event_path):
                if not Example.is_capture_file(filename):
                    continue
                file = open(os.path.join(self.event_path, filename), "r")

                # Get the header line, should be either the first line just after the metadata or the very first line
                line = "#"
                while metadata_regex.match(line):
                    line = file.readline().rstrip("\n")

                # Check each column header for a match
                for col_name in line.split("\t"):
                    for pattern in req_waveforms.keys():
                        if pattern.match(col_name):
                            req_waveforms[pattern] += 1

                # Supposedly garbage collector would take care of this, but this seems cleaner
                file.close()

            # Validate that each of the patterns had exactly one match
            for pattern in req_waveforms.keys():
                if pattern.pattern == "Time":
                    if req_waveforms[pattern] != 8:
                        raise ValueError(
                            "Model found " + str(req_waveforms[pattern]) + " Time columns.  Expected eight.")
                else:
                    if req_waveforms[pattern] > 1:
                        raise ValueError(
                            "Model found multiple waveforms that matched pattern '" + pattern.pattern + "'")
                    if req_waveforms[pattern] < 1:
                        raise ValueError(
                            "Model could not identify require waveform matching pattern '" + pattern.pattern + "'")

    def validate_waveform_times(self, max_start=-100.0, min_end=100.0, step_size=0.2, delta_max=0.02):
        """Verify the Time column of all capture files are identical and have a valid range and sample interval.

        Note: The default 0.02 delta_max is chosen because the actual time step ranges from 0.18... to 0.21... when
        a time step of 0.2 is specified.

            Args:
                max_start (float): The latest acceptable start time for the waveforms
                min_end (float): The earliest acceptable end time for the waveforms
                step_size (float): The expected step_size of each waveform in milliseconds
                delta_max (float): The maximum difference between the observed time steps and step_size in milliseconds.

            Returns:
                None

            Raises:
                ValueError: if either Time columns mismatch or Time columns are beyond expected thresholds

        """

        time = None
        if self.event_df is None:
            # Check that all of the file have the same time series
            first_filename = ""
            for filename in os.listdir(self.event_path):
                if Example.is_capture_file(filename=filename):
                    if time is None:
                        first_filename = filename
                        time = Example.parse_capture_file(os.path.join(self.event_path, filename))['Time']
                    else:
                        if not time.equals(Example.parse_capture_file(os.path.join(self.event_path, filename))['Time']):
                            raise ValueError(
                                "Found Time series mismatch between '{}' and '{}'".format(first_filename, filename))
        else:
            # The DataFrame only has one time field so we can't check that.
            time = self.event_df['Time']

        # Check that the time range is somewhere in the [-1.6s, 1.6s] range
        min_t = min(time)
        max_t = max(time)
        if max_start < min_t or min_end > max_t:
            raise ValueError(
                "Invalid time range of [{},{}] found.  Does not include minimum range for fault data [{}, {}]".format(
                    min_t, max_t, max_start, min_end))

        # Check that the time sample interval is approximately the same.  Since this is floating point, there may be
        # slight differences
        lag = time - time.shift(1)
        lag = lag[1:len(lag)]
        max_step = max(lag)
        min_step = min(lag)
        if abs(step_size - max_step) > delta_max or abs(step_size - min_step) > delta_max:
            raise ValueError("Found improper step size.  Expect: {}, Step size range: ({}, {}), Acceptable delta: {}"
                             .format(step_size, min_step, max_step, delta_max))

    def validate_cavity_modes(self, mode=4, offset=-1.0, deployment='ops'):
        """Checks that each cavity was in the appropriate control mode.

        A request is made to the internal CEBAF myaweb myquery HTTP service at the specified offset from the event
        timestamp.  Currently the proper mode is GDR (I/Q).

        According to the RF low-level software developer (lahti), the proper PV for C100 IOCs is
        R<Linac><Zone><Cavity>CNTL2MODE which is a float treated like a bit word.  At the time of writing, the most
        common modes are:

        * 2 == SEL
        * 4 == GDR (I/Q)

        A single cavity may be bypassed by operations to alleviate performance problems.  In the situation the rest of
        the zone is working normally and is considered to produce valid data for modeling purposes.  Only the control
        modes of the non-bypassed cavities will be considered for invalidating the data.

            Args:
                mode (int):  The mode number associated with the proper control mode.
                offset (float): The number of seconds before the fault event the mode setting should be checked.
                deployment (str): The MYA archiver deployment used for querying historical PV values

            Returns:
                None

            Raises:
                ValueError: if any cavity mode does not match the value specified by the mode parameter.
        """

        # The R???CNTL2MODE PV is a float, treated like a bit word.  GDR (I/Q) mode corresponds to a value of 4.
        mode_template = '{}CNTL2MODE'

        # "Newer" C100 bypass control.  It's a bit word that represents the bypass status of all cavities
        bypassed_template = '{}XMOUT'

        # Still need to check if GSET == 0 since this is how many operators "bypass" a cavity, at least historically.
        gset_template = '{}GSET'

        # Check these PVs just before the fault, since they may have changed in response to the fault
        pre_fault_dt = self.event_datetime + datetime.timedelta(seconds=offset)

        # We need the zone to check the bypass bit word that has bit 0-7 corresponding to cavity 1-8
        zone = None
        for filename in os.listdir(self.event_path):
            if not Example.is_capture_file(filename):
                continue
            zone = filename[0:3]
            break
        if zone is None:
            raise ValueError("Could parse zone name from capture file name")

        # Get the bypassed bit word.  Check each cavity's status in the loop below.
        bypassed = None
        try:
            bypassed = mya.get_pv_value(PV=bypassed_template.format(zone), datetime=pre_fault_dt, deployment=deployment)
        except ValueError:
            # Do nothing here as this bypassed flag was not always archived.  Faults prior to Fall 2019 may predate
            # archival of the R...XMOUT PVs
            pass

        # Switch to binary string.  "08b" means include leading zeros ("0"), have eight bits ("8"), and format string as
        # binary number ("b").  The [::-1] is an extended slice that says to step along the characters in reverse.
        # The reversal puts the bits in cavity order - bit_0 -> cav_1, bit_1 -> cav_2, ...
        bypassed_bits = format(0, "08b")
        if bypassed is not None:
            bypassed_bits = format(bypassed, "08b")[::-1]

        for filename in os.listdir(self.event_path):
            if Example.is_capture_file(filename):
                cav = filename[0:4]

                # Check if the cavity was gset == 0.  Ops meant to bypass this if so, and we don't care about it's
                # control mode
                gset = mya.get_pv_value(PV=gset_template.format(cav), datetime=pre_fault_dt, deployment=deployment)
                if gset == 0:
                    continue

                # Check if the cavity was formally bypassed.  bypassed_bits is zero indexed, while cavities are one
                # indexed.  1 is bypassed, 0 is not
                if bypassed_bits[int(cav[3]) - 1] == 0:
                    continue

                val = mya.get_pv_value(PV=mode_template.format(cav), datetime=pre_fault_dt, deployment=deployment)
                if val != mode:
                    raise ValueError("Cavity '" + cav + "' not in GDR mode.  Mode = " + str(val))

    def validate_zones(self):
        """This method ensures that the model does not make predictions on certain C100 zones, namely 0L04.

            Returns:
                None

            Raises:
                ValueError: if the zone name is 0L04.
        """
        invalid_zones = ['0L04']
        if self.event_zone in invalid_zones:
            raise ValueError("Zone {} is not a valid zone for this model".format(self.event_zone))
