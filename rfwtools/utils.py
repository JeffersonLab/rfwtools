import datetime
import json
import urllib
import requests

from rfwtools.network import SSLContextAdapter


def get_signal_names(cavities, waveforms):
    """Creates a list of signal names by joining each combination of the two lists with _

    Args:
        cavities (list(str)) - A list of strings that represent cavity numbers, e.g. '1' or '7'.  These are the cavities
                               for which signals will be included.
        waveforms (list(str)) - A list of waveform suffixes (e.g., "GMES" or "CRRP") for the waveforms to be included
                                in the output.

    Return list(str) - The list containing all of the combinations of the supplied cavities and waveforms
    """
    signals = []
    for cav in cavities:
        for wf in waveforms:
            signals.append(cav + "_" + wf)
    return signals


def get_events_from_web(data_server="accweb.acc.jlab.org", begin="2018-01-01 00:00:00", end=None):
    """Downloads a a list of events from the waveforms web server which includes only their metadata."""
    if end is None:
        end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    base = 'https://' + data_server + '/wfbrowser/ajax/event?'
    b = urllib.parse.quote_plus(begin)
    e = urllib.parse.quote_plus(end)
    url = base + 'system=rf&out=json&includeData=false' + '&begin=' + b + '&end=' + e

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
