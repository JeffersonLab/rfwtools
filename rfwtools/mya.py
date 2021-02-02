import requests
from rfwtools.network import SSLContextAdapter
from datetime import datetime

__myquery_url__ = "https://myaweb.acc.jlab.org/myquery"
"""str: The base URL of the myquery web service.  Note: this is the internal service URL."""


def get_json(url):
    """Simple function for making an HTTP GET request that should return a valid JSON content-type.

    This method creates a custom SSLContextAdapter that has access to the system's trusts CA certificates.

    Args:
        url (str): The URL on which to perform the HTTP GET

    Returns:
        dict: A dictionary object representing the JSON response

    Raises:
        ValueError: If the URL returns a non-200 status code or if the response is not valid JSON content
    """

    # Setup a custom session that has access to the default set of trusted CA certificates.  The with block closes the
    # session even if their are unhandled exceptions
    with requests.Session() as s:
        adapter = SSLContextAdapter()
        s.mount(url, adapter)
        r = s.get(url)

    if r.status_code != 200:
        raise ValueError(
            "Received error response from {}.  status_code={}.  response={}".format(url, r.status_code, r.text))

    # The built-in JSON decoder will raise a ValueError if parsing non-JSON content
    return r.json()


def get_pv_value(PV, datetime, deployment='ops'):
    """Method for performing a point-type myquery mayweb request.  Returns the only the PV value.

    Args:
        PV (str): The EPICS channel to look up
        datetime (datetime): A datetime object representing the point in time for which the query should be performed
        deployment(str): The name of a valid MYA deployment (defaults to 'ops')

    Returns:
        str: The archived value of PV at datetime according to MYA deployment deployment

    Raises:
        ValueError: If the myquery point service returns an error response.
    """
    timestamp = datetime.strftime("%Y-%m-%d+%H:%M:%S.%f")
    query = "/point?c={}&t={}&m={}&f=&v=".format(PV, timestamp, deployment)
    json = get_json(__myquery_url__ + query)

    # Shouldn't happen since make_json_request checks for status_code == 200
    if 'error' in json.keys():
        raise ValueError("Received error response - {}".format(json['error']))

    # Possible that there is no data for the time queried (e.g., the time is before we started archiving that PV)
    out = None
    data = json['data']
    if 'v' in data.keys():
        out = data['v']

    return out
