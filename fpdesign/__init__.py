"""fpdesign: the repo's shared FP design/modelling library.

  * :mod:`fpdesign.peak_models` -- ESM-2 embedding utilities and the surrogate/oracle
    architectures (``build_base``, pooling readouts, CNN/Transformer heads). Also imported
    outside the campaigns, by ``dataset_pipeline/`` and ``GFP_DMS/``, for the ESM-2 helpers;
  * :mod:`fpdesign.pockets` -- structure-based chromophore-pocket rules (5 A edit windows,
    H-bond partners), read against the repo-level ``structures/`` PDBx cache;
  * :mod:`fpdesign.campaign` -- the parameterized design-campaign engine (guided + gibbs strategies);
  * ``fpdesign/models/`` -- the shared surrogate + brightness checkpoints;
  * ``fpdesign/build_design_windows.py`` -- structure-based edit-window generation.

Callers put the REPO ROOT on ``sys.path`` and import through the package::

    sys.path.insert(0, str(REPO))
    from fpdesign import peak_models as pm

``in-silico-test/`` deliberately does NOT import from here -- it vendors its own copies under
``in-silico-test/lib/`` to stay self-contained.
"""
from .campaign import Campaign, CampaignConfig, load_dataset, run, DEFAULT_SURROGATE

__all__ = ["Campaign", "CampaignConfig", "load_dataset", "run", "DEFAULT_SURROGATE"]
