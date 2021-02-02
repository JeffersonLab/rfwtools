import datetime
from unittest import TestCase
from rfwtools.timestamp import TimestampMapper, is_datetime_in_range

fmt = "%Y-%m-%d %H:%M:%S.%f"
dt = datetime.datetime.strptime("2020-01-01", "%Y-%m-%d")
dt0 = datetime.datetime.strptime("2018-01-01", "%Y-%m-%d")
dt1 = datetime.datetime.strptime("2019-01-01", "%Y-%m-%d")
dt2 = datetime.datetime.strptime("2019-06-01", "%Y-%m-%d")
dt3 = datetime.datetime.strptime("2020-01-01", "%Y-%m-%d")
dt4 = datetime.datetime.strptime("2020-01-02", "%Y-%m-%d")
dt5 = datetime.datetime.strptime("2021-01-01", "%Y-%m-%d")


class TestTimestamp(TestCase):

    def test_update_timestamp_map(self):
        # This is a weak test, but it shows that something is being downloaded

        # Make an instance and blank it's internal map
        tsm = TimestampMapper()
        TimestampMapper._label_to_database_timestamp_map = None

        # Update the map and check that some data is present
        tsm.update_timestamp_map()
        self.assertIsNotNone(tsm._label_to_database_timestamp_map)
        self.assertIn('1L22', tsm._label_to_database_timestamp_map.keys())

    def test_get_full_timestamp(self):
        # Test that we can get a result for a known event

        tsm = TimestampMapper()
        expected = datetime.datetime.strptime("2018-08-01 17:02:52.9", fmt)
        result = tsm.get_full_timestamp('1L23', expected.replace(microsecond=0))

        self.assertEqual(expected, result)

    def test__generate_timestamp_map(self):
        # Test that we can successfully generate a valid timestamp map

        self.maxDiff = None
        # Make sure to manually just for the UTC offset in the expected output
        expected = {
            '1L23': {
                datetime.datetime.strptime("2018-08-01 13:02:52.0", fmt): datetime.datetime.strptime(
                    "2018-08-01 13:02:52.9", fmt),
                datetime.datetime.strptime("2018-08-01 13:04:02.0", fmt): datetime.datetime.strptime(
                    "2018-08-01 13:04:02.9", fmt)
            },
            '1L24': {
                datetime.datetime.strptime("2018-08-04 20:03:03.0", fmt): datetime.datetime.strptime(
                    "2018-08-04 20:03:03.9", fmt)
            }
        }

        event_list = [
            {'location': '1L23', 'datetime_utc': "2018-08-01 17:02:52.9"},
            {'location': '1L23', 'datetime_utc': "2018-08-01 17:04:02.9"},
            {'location': '1L24', 'datetime_utc': "2018-08-05 00:03:03.9"}
        ]
        result = TimestampMapper._generate_timestamp_map(event_list)

        self.assertDictEqual(expected, result)

    def test_is_datetime_in_range_matches(self):
        # Test that it matches when it should
        # Match cases
        r1 = [[None, None]]  # Matches - All time
        r2 = [[dt2, dt3], [dt4, dt5]]  # Matches - end inclusive
        r3 = [[dt1, dt2], [dt3, dt4]]  # Matches - start inclusive
        r4 = [[dt2, dt4], [dt3, dt4]]  # Matches - match internal to range
        r5 = [[None, dt3]]  # Matches - Inf start
        r6 = [[dt1, None]]  # Matches - Inf end

        self.assertTrue(is_datetime_in_range(dt, r1), "all time - match")
        self.assertTrue(is_datetime_in_range(dt, r2), "end inclusive - match")
        self.assertTrue(is_datetime_in_range(dt, r3), "start inclusive - match")
        self.assertTrue(is_datetime_in_range(dt, r4), "internal - match")
        self.assertTrue(is_datetime_in_range(dt, r5), "inf start - match")
        self.assertTrue(is_datetime_in_range(dt, r6), "inf end - match")

    def test_is_datetime_in_range_no_match(self):
        # Test that it does not match when it should not
        r1 = [[None, dt1]]  # No match - Inf start
        r2 = [[dt5, None]]  # No match - Inf end
        r3 = [[dt1, dt2]]  # No match - single
        r4 = [[dt1, dt2], [dt4, dt5]]  # No match - multiple
        r5 = None  # No match - None provided as range

        self.assertFalse(is_datetime_in_range(dt, r1), "inf start - no match")
        self.assertFalse(is_datetime_in_range(dt, r2), "inf end - no match")
        self.assertFalse(is_datetime_in_range(dt, r3), "single - no match")
        self.assertFalse(is_datetime_in_range(dt, r4), "multiple - no match")
        self.assertFalse(is_datetime_in_range(dt, r5), "None range - no match")

    def test_is_datetime_in_range_exceptions(self):
        # Test that it throws exceptions as expected
        r1 = [['asdf', dt1]]  # Invalid value
        r2 = [[[dt1, dt2], [dt4, dt3]]]  # Out of order
        r3 = [[dt1]]  # Single value, not a range

        # Have to use the with syntax to use function inline.  Otherwise, they want a callable, not the return value of
        # the function that is going to throw an exception
        with self.assertRaises(ValueError):
            is_datetime_in_range(dt, r1)
        with self.assertRaises(ValueError):
            is_datetime_in_range(dt, r2)
        with self.assertRaises(IndexError):
            is_datetime_in_range(dt, r3)
