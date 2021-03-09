"""This module provides commonly needed functionality for interacting with timestamps."""

from datetime import datetime
import os
import pickle
from typing import Dict, List, Tuple

import tzlocal

from rfwtools import utils
from rfwtools.config import Config


class TimestampMapper:
    """Class for mapping timestamps in Tom's label files to the timestamp with fractional seconds used elsewhere."""

    #: (Dict[str, Dict[datetime, datetime]]): Single class-wide cache of label to database timestamp mappings.
    _label_to_database_timestamp_map = None

    def get_full_timestamp(self, zone: str, dt: datetime) -> str:
        """Returns the full timestamp based on the supplied zone and timestamp strings.  Expects label file format

        Args:
            zone: The fault event's zoneformat is CED (e.g., 1L23).
            dt: The datetime object containing the truncated time.  Typical that microseconds = 0 (Tom's format).

        Returns:
              Standard web-based format timestamp string, e.g., "2019-02-01 01:15:30.2"
        """

        # Get the timestamp map if we don't already have one
        if TimestampMapper._label_to_database_timestamp_map is None:
            self.update_timestamp_map()

        # Check if we have the needed keys.  If not, raise a known exception that can be caught
        if zone not in TimestampMapper._label_to_database_timestamp_map.keys():
            raise ValueError(f"zone '{zone}' not found in  timestamp mapper")
        if dt not in TimestampMapper._label_to_database_timestamp_map[zone].keys():
            raise ValueError(f"event '{zone} / {dt}' not found in  timestamp mapper")

        return TimestampMapper._label_to_database_timestamp_map[zone][dt]

    @staticmethod
    def update_timestamp_map(mapper: Dict[str, Dict[datetime, datetime]] = None) -> None:
        """Creates/replaces a nested dict mapping event timestamps without fractional seconds to those with fractions

        Args:
            mapper: If none, one is generated.  Otherwise, the given map replaces the existing one.
        """
        if mapper is None:
            results = utils.get_events_from_web()
            TimestampMapper._label_to_database_timestamp_map = TimestampMapper._generate_timestamp_map(
                results['events'])
        else:
            TimestampMapper._label_to_database_timestamp_map = mapper

    @staticmethod
    def _generate_timestamp_map(event_list: List[Dict]) -> Dict[str, Dict[datetime, datetime]]:
        """Generate a dictionary that maps a label file event to its full timestamp.

        Args:
            event_list: A list of dictionaries describing events.  These events are expected to have the format of a web
                response (/wfbrowser/ajax/events).  E.g.:
                [
                    {
                      "id": 119239,
                      "datetime_utc": "2020-03-01 23:15:36.3",
                      "location": "2L24",
                      "system": "rf",
                      "archive": false,
                      "classification": "",
                      "captureFiles": [],
                      "labels": null
                    }, ...
                ]


        Returns:
            A two-level deep dictionary where every entry represents an event in the wfbrowser database.  First
            keyed on zone strings, then on datetimes with microsecond == 0.  Values are the same datetime with the
            microseconds value of the database record for that event.

            Looks like this at the end.
            {
              '1L07': {
                        <datetime1 w/o microseconds>: <datetime1 w/ microseconds>,
                        <datetime2 w/o microseconds>: <datetime2 w/ microseconds>,
                         ..,
              },
              '1L21": {...},
              ...
            }

        """

        event_timestamp_map = dict()
        for event in event_list:
            # Get a timezone aware datetime object of UTC timestamp (manually add GMT offset string) then convert it
            # to local time
            dt_local = datetime.strptime(event['datetime_utc'] + "-0000",
                                                  '%Y-%m-%d %H:%M:%S.%f%z').astimezone(tzlocal.get_localzone())
            dt_local = dt_local.replace(tzinfo=None)

            zone = event['location']
            if zone not in event_timestamp_map.keys():
                event_timestamp_map[zone] = dict()

            # Store a mapping between datetime objects. w/o fraction to w/ faction.  Note: datetime.replace makes a copy
            event_timestamp_map[zone][dt_local.replace(microsecond=0)] = dt_local

        return event_timestamp_map

    def save_mapper(self, filename: str) -> None:
        """Method to save the mapper to pickle file.  Relative to Config().output_dir, unless path is absolute.

        Arguments:
            filename: The name of the file.  Relative to Config().output_dir unless path is absolute.
        """

        if not os.path.isabs(filename):
            file = os.path.join(Config().output_dir, filename)

        with open(filename, "wb") as f:
            pickle.dump(TimestampMapper._label_to_database_timestamp_map, f)


def is_datetime_in_range(dt: datetime, range_list: List[Tuple[datetime, datetime]]) -> bool:
    """Check if the supplied datetime object is in any of the specified ranges.

    Note:  If you are giving a single range, use a list, not a tuple.  Seems tuples are reduced to the single inner list
    if only one element is supplied.

    Args:
        dt:
            A datetime object to compare
        range_list:
            A list of 2-tuples of datetime's. Each pair describes a time range for which dt may be in.  Ranges are
            inclusive on both ends.  Leaving an end point in a range as None is treated though it were infinity.

    Returns:
        True if dt is an any of the supplied ranges.  False otherwise.
    """

    # Make sure we got the right input type
    if type(dt) != datetime:
        raise ValueError("dt must be of type datetime")

    # Can't be in a range that does not exist
    if range_list is None:
        return False

    # Work through the range_list and see if dt is in any of them
    in_range = False
    for d_range in range_list:
        start = d_range[0]
        end = d_range[1]

        # Check that we received datetime objects (or None)
        if start is not None and type(start) != datetime:
            raise ValueError("date ranges may only include None or type datetime")
        if end is not None and type(end) != datetime:
            raise ValueError("date ranges may only include None or type datetime")

        # Check if it is in the range.  Treat None as +/- Inf.
        if start is None and end is None:
            in_range = True
            break
        elif start is None and end is not None:
            if dt <= end:
                in_range = True
                break
        elif end is None and start is not None:
            if start <= dt:
                in_range = True
                break
        else:
            # Make sure that the order is sensible
            if end < start:
                raise ValueError("Individual ranges must be given as [start, end], with start <= end")

            # Normal range check
            if start <= dt <= end:
                in_range = True
                break

    return in_range
