"""Remarks -> condition score and motivation flags.

The phrase dictionary (tag_remarks.py) finds condition phrases and motivation wording.
A latent-class model (condition_lca.py, parameters in assets/) turns the condition
phrases into a 1-5 level; remarks with fewer than two condition phrases fall back to a
small TF-IDF text model (assets/condition_fallback.json). Neither model ever saw a price.

Results are cached by ml_num, so each remark is processed once per run.
"""
import json
import os
import warnings

import numpy as np
import pandas as pd

from . import ASSETS
from condition_lca import apply as lca_apply, combine, text_probs
from tag_remarks import tag

MOTIVATION = ["mot_any", "mot_offers", "mot_quick_close", "mot_flexible_close"]
COLUMNS = ["cond_exp", "cond_class"] + MOTIVATION

_params = json.load(open(os.path.join(ASSETS, "condition_lca_params.json")))
_fallback = json.load(open(os.path.join(ASSETS, "condition_fallback.json")))
_cache = pd.DataFrame(columns=COLUMNS, dtype=float)


def remark_features(homes):
    """Return condition and motivation columns for each row of `homes` (same order)."""
    global _cache
    todo = homes.loc[~homes.ml_num.isin(_cache.index), ["ml_num", "remarks"]].drop_duplicates("ml_num")
    if len(todo):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tags = tag(todo.remarks.reset_index(drop=True))
        tags.insert(0, "ml_num", todo.ml_num.values)
        cond = combine(tags, lca_apply(tags, _params), text_probs(todo.remarks.fillna("").tolist(), _fallback))
        new = pd.DataFrame({"cond_exp": cond.cond_exp.values, "cond_class": cond.cond_class.values,
                            **{m: tags[m].values for m in MOTIVATION}}, index=todo.ml_num.values)
        _cache = pd.concat([_cache, new]) if len(_cache) else new
    return _cache.loc[homes.ml_num.values, COLUMNS].reset_index(drop=True).set_index(homes.index)
