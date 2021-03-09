"""This module contains general utility functions that may be used throughout the package."""

import datetime
import json
import urllib
from typing import List

import requests

from rfwtools.config import Config
from rfwtools.network import SSLContextAdapter


def get_signal_names(cavities: List[str], waveforms: List[str]) -> List[str]:
    """Creates a list of signal names by joining each combination of the two lists with _

    Args:
        cavities:
            A list of strings that represent cavity numbers, e.g. '1' or '7'.  These are the cavities for which signals
            will be included.
        waveforms:
            A list of waveform suffixes (e.g., "GMES" or "CRRP") for the waveforms to be included in the output.

    Return:
        The list containing all of the combinations of the supplied cavities and waveforms
    """
    signals = []
    for cav in cavities:
        for wf in waveforms:
            signals.append(cav + "_" + wf)
    return signals


def get_events_from_web(data_server: str = None, wfb_base_url: str = None, begin: str = "2018-01-01 00:00:00",
                        end: str = None) -> dict:
    """Downloads a a list of events from the waveforms web server which includes only their metadata.

    Arguments:
        data_server: The hostname of the service running the waveform browser.  Defaults to Config().data_server.
        wfb_base_url: The base string of the URL for the waveform browser.  Defaults to Config().wfb_base_url.
        begin: A string formatted "%Y-%m-%d %H:%M:%S" that marks the beginning of the requested range.  Defaults to a
            date well before the first harvester files were captured.
        end: A string formatted "%Y-%m-%d %H:%M:%S" that marks the end of the requested range.

    Returns:
        The JSON response converted to Python data structures.  Outer structure is expected to be a dictionary.
    """

    if data_server is None:
        data_server = Config().data_server
    if wfb_base_url is None:
        wfb_base_url = Config().wfb_base_url

    if end is None:
        end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    base = f'https://{data_server}/{wfb_base_url}/ajax/event?'
    b = urllib.parse.quote_plus(begin)
    e = urllib.parse.quote_plus(end)
    url = f'{base}system=rf&out=json&includeData=false&begin={b}&end={e}'

    # Download the metadata about all of the events - supply the session/SSLContextAdapter to use system trust store
    # (required for Windows use)
    s = requests.Session()
    adapter = SSLContextAdapter()
    s.mount(url, adapter)
    r = s.get(url)

    # Test if we got a good status code.
    if not r.status_code == 200:
        raise RuntimeError(f"Received non-ok response - {r.status_code}.  url={url}")

    return json.loads(r.content)
