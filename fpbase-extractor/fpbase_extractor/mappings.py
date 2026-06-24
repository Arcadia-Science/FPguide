"""Human-readable labels for FPbase coded enum fields.

FPbase stores oligomerization state and switching behaviour as short codes.
These maps expand them to readable labels. Codes come from the FPbase data
model; lookups are case-insensitive and fall back to the raw code.
"""

# Oligomerization / aggregation tendency (`agg` field).
AGG_LABELS = {
    "m": "monomer",
    "d": "dimer",
    "wd": "weak dimer",
    "td": "tandem dimer",
    "t": "tetramer",
}

# Photoswitching behaviour (`switch_type` field).
SWITCH_TYPE_LABELS = {
    "b": "basic",
    "pa": "photoactivatable",
    "ps": "photoswitchable",
    "pc": "photoconvertible",
    "ts": "timer",
    "mp": "multistate",
    "o": "other",
}


def _lookup(table, code):
    if code is None:
        return ""
    key = str(code).strip().lower()
    if not key:
        return ""
    return table.get(key, key)


def oligomerization(code):
    """Return a readable oligomerization label for an `agg` code."""
    return _lookup(AGG_LABELS, code)


def switch_type(code):
    """Return a readable label for a `switch_type` code."""
    return _lookup(SWITCH_TYPE_LABELS, code)
