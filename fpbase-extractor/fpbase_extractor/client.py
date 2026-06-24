"""Client for fetching fluorescent protein data from FPbase (https://www.fpbase.org).

The whole database (~1000 proteins) is small enough to retrieve in a single
GraphQL request, which also returns the richest data (sequence, every
photophysical state, parent organism, and primary reference). A REST fallback
is provided in case the GraphQL endpoint is unavailable.

No third-party dependencies — uses urllib from the standard library.
"""

import json
import urllib.error
import urllib.request

GRAPHQL_URL = "https://www.fpbase.org/graphql/"
REST_URL = "https://www.fpbase.org/api/proteins/"
SPECTRA_URL = "https://www.fpbase.org/api/proteins/spectra/"

# A browser-like UA: FPbase sits behind a filter that rejects default urllib UAs.
_USER_AGENT = "fpbase-extractor/1.0 (+https://www.fpbase.org/api/)"

# Single query that pulls every protein with all the sequence and phenotype
# data FPbase exposes. `states` is the per-conformation photophysical data
# (a protein may have several states, e.g. photoswitchable on/off forms).
_BULK_QUERY = """
{
  proteins {
    name
    slug
    seq
    aliases
    genbank
    uniprot
    ipgId
    pdb
    agg
    switchType
    cofactor
    parentOrganism { scientificName }
    primaryReference { doi year journal title }
    states {
      name
      slug
      exMax
      emMax
      extCoeff
      qy
      brightness
      pka
      maturation
      lifetime
      twopExMax
      twopPeakGm
      twopQy
      isDark
      emhex
      exhex
    }
  }
}
"""


class FPbaseError(RuntimeError):
    """Raised when FPbase returns an error or unexpected payload."""


def _post_json(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": _USER_AGENT},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_graphql(timeout=60):
    """Fetch all proteins via the GraphQL bulk query.

    Returns a list of protein dicts using camelCase keys as returned by the API.
    """
    try:
        result = _post_json(GRAPHQL_URL, {"query": _BULK_QUERY}, timeout)
    except urllib.error.URLError as exc:
        raise FPbaseError(f"GraphQL request failed: {exc}") from exc

    if "errors" in result and result["errors"]:
        raise FPbaseError(f"GraphQL returned errors: {result['errors']}")
    proteins = (result.get("data") or {}).get("proteins")
    if proteins is None:
        raise FPbaseError("GraphQL response missing data.proteins")
    return proteins


def fetch_rest(timeout=60):
    """Fetch all proteins via the REST endpoint (fallback data source).

    Returns a list of protein dicts using the REST field names (snake_case,
    a subset of the GraphQL fields — no organism/reference detail or aliases).
    """
    url = f"{REST_URL}?format=json"
    try:
        return _get_json(url, timeout)
    except urllib.error.URLError as exc:
        raise FPbaseError(f"REST request failed: {exc}") from exc


def fetch_spectra(timeout=120):
    """Fetch full excitation/emission spectra for all proteins that have them.

    Uses the dedicated REST spectra endpoint, which returns every protein's
    spectral curves in a single request. Each protein dict has a `spectra` list;
    each spectrum has `state` (e.g. "default_ex"), `ec`, `max`, and `data`
    (a list of [wavelength, intensity] pairs).
    """
    url = f"{SPECTRA_URL}?format=json"
    try:
        return _get_json(url, timeout)
    except urllib.error.URLError as exc:
        raise FPbaseError(f"Spectra request failed: {exc}") from exc


def fetch_proteins(source="graphql", timeout=60):
    """Fetch all proteins.

    source:
        "graphql" — richest data, single request (default).
        "rest"    — REST endpoint, fewer fields.
        "auto"    — try GraphQL, fall back to REST on failure.

    Returns a tuple of (proteins, source_used).
    """
    if source == "rest":
        return fetch_rest(timeout), "rest"
    if source == "graphql":
        return fetch_graphql(timeout), "graphql"
    if source == "auto":
        try:
            return fetch_graphql(timeout), "graphql"
        except FPbaseError:
            return fetch_rest(timeout), "rest"
    raise ValueError(f"unknown source: {source!r}")
