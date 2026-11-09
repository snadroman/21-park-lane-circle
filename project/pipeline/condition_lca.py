"""Condition level (1-5) from the remark's condition phrases, plus the text fallback.

Latent-class model: a mixture of 5 classes of independent Bernoullis over the 69 condition phrases
from tag_remarks.py, fitted by EM. Classes are ordered worst -> best (needs work / dated /
original but good / updated / renovated) by the average tier of the phrases they use.
Text fallback: TF-IDF + logistic regression (trained by tools/train_fallback.py), run here in
numpy. combine() uses the latent-class probabilities when a remark has at least MIN_HARD tiered
phrases and the fallback's otherwise.

  fit:    python pipeline/condition_lca.py fit   remarks_tags.csv condition_lca_params.json
  apply:  python pipeline/condition_lca.py apply remarks_tags.csv condition_lca_params.json condition_lca.csv
"""
import json
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

from tag_remarks import ATOMS

# inputs: the 66 tiered phrases and the three untiered condition signals
COLS = [f"a_{n}" for n, (t, _, _) in ATOMS.items() if t] + ["a_as_is", "a_good_bones", "a_renovator_pitch"]
NAMES = COLS  # kept for the params file
TIER = {f"a_{n}": t for n, (t, _, _) in ATOMS.items() if t}


def _matrix(df):
    return df[COLS].values.astype(float)


def _posterior(X, pi, th):
    ll = X @ np.log(th).T + (1 - X) @ np.log(1 - th).T + np.log(pi)
    m = ll.max(1, keepdims=True)
    R = np.exp(ll - m)
    s = R.sum(1, keepdims=True)
    return R / s, float((m + np.log(s)).sum())


def fit(X, K=5, seeds=3, iters=300):
    best = None
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        R = rng.dirichlet(np.ones(K), len(X))
        for _ in range(iters):
            pi = R.mean(0)
            th = np.clip((R.T @ X) / R.sum(0)[:, None], 1e-4, 1 - 1e-4)
            R, ll = _posterior(X, pi, th)
        if best is None or ll > best[0]:
            best = (ll, pi, th)
    _, pi, th = best
    # order classes worst -> best by the mean tier of atoms each class emits
    tiers = np.array([TIER.get(n, np.nan) for n in NAMES])
    ok = ~np.isnan(tiers)
    score = (th[:, ok] * tiers[ok]).sum(1) / th[:, ok].sum(1)
    order = np.argsort(score)
    return pi[order], th[order]


def apply(df, params):
    pi, th = np.array(params["pi"]), np.array(params["theta"])
    R, _ = _posterior(_matrix(df), pi, th)
    out = pd.DataFrame({"ml_num": df["ml_num"].values})
    out["cond_lca_class"] = R.argmax(1) + 1
    out["cond_lca_exp"] = (R * np.arange(1, R.shape[1] + 1)).sum(1).round(4)
    for k in range(R.shape[1]):
        out[f"cond_lca_p{k + 1}"] = R[:, k].round(4)
    out["cond_lca_conf"] = R.max(1).round(4)
    return out


_TOKEN = re.compile(r"(?u)\b\w\w+\b")  # same tokenizer as sklearn's TfidfVectorizer


def text_probs(texts, model):
    """numpy re-implementation of the trained TF-IDF + logistic-regression fallback."""
    index = {t: i for i, t in enumerate(model["terms"])}
    idf = np.array(model["idf"])
    W, b = np.array(model["coef"]), np.array(model["intercept"])
    P = np.zeros((len(texts), len(b)))
    for r, s in enumerate(texts):
        toks = _TOKEN.findall(s.lower().replace("s/s", "stainless steel"))
        grams = toks + [f"{a} {c}" for a, c in zip(toks, toks[1:])]
        tf = Counter(index[g] for g in grams if g in index)
        logits = b.copy()
        if tf:
            ids = np.fromiter(tf.keys(), int)
            v = (1 + np.log(np.fromiter(tf.values(), float))) * idf[ids]
            v /= np.linalg.norm(v)
            logits += W[:, ids] @ v
        e = np.exp(logits - logits.max())
        P[r] = e / e.sum()
    return P


MIN_HARD = 2


def combine(df, lca, P_text):
    """Final condition: latent-class probabilities when the remark has at least MIN_HARD tiered
    condition phrases, otherwise the text fallback's."""
    has = df["cond_n_atoms"].values >= MIN_HARD
    R_lca = lca[[f"cond_lca_p{k}" for k in range(1, 6)]].values
    R = np.where(has[:, None], R_lca, P_text)
    out = pd.DataFrame({"ml_num": df["ml_num"].values})
    out["cond_class"] = R.argmax(1) + 1
    out["cond_exp"] = (R * np.arange(1, 6)).sum(1).round(4)
    for k in range(5):
        out[f"cond_p{k + 1}"] = R[:, k].round(4)
    out["cond_source"] = np.where(has, "dictionary", "text_fallback")
    return out


if __name__ == "__main__":
    mode, tags = sys.argv[1], pd.read_csv(sys.argv[2])
    if mode == "fit":
        pi, th = fit(_matrix(tags))
        json.dump({"atoms": NAMES, "pi": pi.tolist(), "theta": th.tolist()}, open(sys.argv[3], "w"))
        print("class shares (worst->best):", np.round(pi, 3))
    else:
        params = json.load(open(sys.argv[3]))
        assert params["atoms"] == NAMES, "phrase list changed since fitting; refit"
        res = apply(tags, params)
        res.to_csv(sys.argv[4], index=False)
        print(res.cond_lca_class.value_counts(normalize=True).sort_index().round(3).to_dict())
