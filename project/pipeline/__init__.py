"""Our pipeline, used by models.py.

    remarks.py   remarks -> condition score (1-5) and seller-motivation flags
    market.py    monthly price and sold/list indexes per property type, and their forecast
    homes.py     per-home features and the LightGBM models on top of the indexes

The remark tagger and condition model (tag_remarks.py, condition_lca.py) and the
forecasters (forecast_classical.py, tft_forecast.py) are kept as standalone modules, so
this folder is put on the import path.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
if HERE not in sys.path:
    sys.path.insert(0, HERE)


SEED = 20261028
THREADS = 2      # fixed so results don't depend on how many cores the machine has


def make_deterministic():
    """Seed every random number generator the pipeline touches and fix thread counts.
    Called at the start of every fit. (LightGBM and the TFT also get explicit seeds.)"""
    import random
    import numpy as np
    random.seed(SEED)
    np.random.seed(SEED)
    try:
        import torch
        torch.manual_seed(SEED)
        torch.set_num_threads(THREADS)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
