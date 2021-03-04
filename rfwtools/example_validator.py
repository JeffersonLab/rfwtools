"""This package manages the validation of Examples.  While much data is collected, some data is unsuitable for analysis.

An ExampleValidator object is responsible for investigating an Example object and determining if it suitable for
analysis.  ExampleValidator should be subclassed as needed to support the individual requirements of different analytical approaches.

Typically you won't use these directly, and instead pass it to the ExapmleSet.purge_invalid_examples() method.
Basic Usage Example:
::

    import math
    from datetime import datetime
    from rfwtools.example import Example
    from rfwtools.example_validator import ExampleValidator

    # Make an example to validate
    ex = Example(zone='1L25',
                 dtime=datetime.strptime("2020-03-10 01:08:41.2", "%Y-%m-%d %H:%M:%S.%f),
                 cavity_label="4",
                 fault_label="Microphonics",
                 cavity_conf=math.nan,
                 fault_conf=math.nan,
                 label_source='my_label_file.txt'
                )

    # Setup the validator
    ev = ExampleValidator()
    ev.set_example(ex)

    # If anything is wrong with the example, the validator will raise an exception.  The exception clause is
    # intentionally broad to capture the bevy of problems that could be encountered.
    try:
        ev.validate_data()
    except Exception as exc:
        print(f"Invalid event - {ex}\\n  {exc}\\n")

"""

import datetime
import itertools

from rfwtools import mya
from rfwtools.example import Example


class ExampleValidator:
    """This class provides functionality for checking that an individual example meets the criteria for validity.

    Some checks are very basic, e.g., do we have all of the necessary data from the fault event.  Others are a bit more
    nuanced, e.g., was the cavity in the proper RF mode. See validate_data and other validation methods for details.

    Note:
        This class loads capture file data at set_example, but defers any exceptions from that process until
        validate_data() is called.
    """

    def __init__(self):
        """Create an instance for validating Example."""
        #: (datetime): The datetime of the fault.
        self.event_datetime = None
        #: (str): The zone where the fault occurred
        self.event_zone = None
        #: (pd.DataFrame): The DataFrame of waveform signals
        self.event_df = None
        #: (dict of str:str): The raw capture file content (typically produced by the harvester daemon)
        self.event_cf_content = None

    def set_example(self, example: Example) -> None:
        """Set internal information about the example to validate.

        Arguments:
            example: The example that is to be validated.
        """
        self.event_cf_content = example.get_capture_file_contents()
        self.event_datetime = example.event_datetime
        self.event_zone = example.event_zone

        # Need to load the data and make an internal copy for later use.
        try:
            example.load_data()
            self.event_df = example.event_df.copy()
            example.unload_data()
        except Exception:
            self.event_df = None

    def validate_data(self, deployment: str = 'ops') -> None:
        """Check that the event directory and it's data is of the expected format.

        This method inspects the event directory and raises an exception if a problem is found.  The following aspects
        of the event directory and waveform data are validated.

        * Data can be found on disk

        * All eight cavities are represented by exactly one capture file

        * All of the required waveforms are represented exactly once

        * All of the capture files use the same timespan and have constant sampling intervals

        * All of the cavity are in the appropriate control mode (GDR I/Q => 4) or bypassed

        Arguments:
            deployment: Which MYA deployment should be used when checking archiver data.

        Raises:
            ValueError: If a problem is found with the data.

        """
        if self.event_df is None:
            raise ValueError("Error getting event_df during set_example()")
        self.validate_capture_file_counts()
        self.validate_capture_file_waveforms()
        self.validate_waveform_times()
        self.validate_cavity_modes(deployment=deployment)
        self.validate_zones()

    def validate_capture_file_counts(self) -> None:
        """This method checks that we have exactly one capture file per cavity/IOC.

        The harvester grouping logic coupled with unreliable IOC behavior seems to produce fault event directories where
        either an IOC has multiple capture files or are missing.  We want to make sure we have exactly eight capture
        files - one per IOC.  Raises an exception in the case that something is amiss.

        Raises:
            ValueError: if either missing or "duplicate" capture files are found.
        """

        # Count capture files per cavity
        capture_file_counts = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0}

        # Count the filenames that match each cavity
        for filename in self.event_cf_content.keys():
            cavity = filename[3]
            capture_file_counts[cavity] += 1

        # Verify that we have the right counts - one for each cavity
        for cavity in capture_file_counts.keys():
            if capture_file_counts[cavity] > 1:
                raise ValueError("Duplicate capture files exist for zone '" + cavity + "'")
            if capture_file_counts[cavity] == 0:
                raise ValueError("Missing capture file for zone '" + cavity + "'")

    def validate_capture_file_waveforms(self) -> None:
        """Checks that all of the required waveforms are present exactly one time across all capture files.

        If event_df is None, then the capture files themselves are loaded.  If event_df is not None, then the files are
        checked directly.

        Raises:
            ValueError: if any required waveform is repeated or missing
        """

        # Get a structure for counting matches of waveforms
        req_signals = ["IMES", "QMES", "GMES", "PMES", "IASK", "QASK", "GASK", "PASK", "CRFP", "CRFPP",
                       "CRRP", "CRRPP", "GLDE", "PLDE", "DETA2", "CFQE2", "DFQES"]

        # Assume we have the event_df.  Compare the DataFrame columns to the required signals
        if self.event_df is None:
            raise ValueError("Missing fault event waveform data (event_df)")

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

    def validate_waveform_times(self, max_start: float = -100.0, min_end: float = 100.0, step_size: float = 0.2,
                                delta_max: float = 0.02) -> None:
        """Verify the Time column of all capture files are identical and have a valid range and sample interval.

        Note: The default 0.02 delta_max is chosen because the actual time step ranges from 0.18... to 0.21... when
        a time step of 0.2 is specified.

        Arguments:
            max_start: The latest acceptable start time for the waveforms
            min_end: The earliest acceptable end time for the waveforms
            step_size: The expected step_size of each waveform in milliseconds
            delta_max: The maximum difference between the observed time steps and step_size in milliseconds.

        Raises:
            ValueError: if either Time columns mismatch or Time columns are beyond expected thresholds
        """

        if self.event_df is None:
            raise ValueError("Missing fault event waveform data (event_df)")

        # Check that the time range is somewhere in the [-1.6s, 1.6s] range
        time = self.event_df['Time']
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

    def validate_cavity_modes(self, mode: int = 4, offset: float = -1.0, deployment: str = 'ops') -> None:
        """Checks that each cavity was in the appropriate control mode or is bypassed.

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

        Arguments:
            mode:  The mode number associated with the proper control mode.
            offset: The number of seconds before the fault event the mode setting should be checked.
            deployment: The MYA archiver deployment used for querying historical PV values

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
        cfs = list(self.event_cf_content.keys())
        if len(cfs) > 0:
            filename = cfs[0]
            zone = filename[0:3]
        else:
            raise ValueError("No capture file content found.")

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

        for filename in self.event_cf_content.keys():
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

    def validate_zones(self) -> None:
        """This method ensures that the model does not make predictions on certain C100 zones, namely 0L04.

        Raises:
            ValueError: if the zone name is 0L04.
        """
        invalid_zones = ['0L04']
        if self.event_zone in invalid_zones:
            raise ValueError("Zone {} is not a valid zone for this model".format(self.event_zone))
