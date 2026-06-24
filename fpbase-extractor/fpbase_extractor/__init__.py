"""fpbase_extractor — extract protein sequences and phenotype data from FPbase."""

__version__ = "1.0.0"

from .client import FPbaseError, fetch_proteins, fetch_spectra
from .extract import normalize_all, normalize_protein, write_outputs
from .spectra import normalize_spectra, write_spectra_outputs

__all__ = [
    "FPbaseError",
    "fetch_proteins",
    "fetch_spectra",
    "normalize_protein",
    "normalize_all",
    "write_outputs",
    "normalize_spectra",
    "write_spectra_outputs",
    "__version__",
]
