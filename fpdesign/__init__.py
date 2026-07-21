"""fpdesign: shared library for the conventional FP design campaigns.

Common assets referenced across the per-campaign folders under ``design-campaign-conventional/``:
  * :mod:`fpdesign.campaign` -- the parameterized design-campaign engine (guided + gibbs strategies);
  * ``fpdesign/models/`` -- the shared surrogate checkpoint;
  * ``fpdesign/build_design_windows.py`` -- structure-based edit-window generation.
"""
from .campaign import Campaign, CampaignConfig, load_dataset, run, DEFAULT_SURROGATE

__all__ = ["Campaign", "CampaignConfig", "load_dataset", "run", "DEFAULT_SURROGATE"]
