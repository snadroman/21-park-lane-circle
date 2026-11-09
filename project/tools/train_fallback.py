"""Train the condition text fallback: TF-IDF (word 1-2-grams) + multinomial logistic regression,
taught to predict the latent condition class from remark text. Exported to JSON so the pipeline
runs it with numpy only (condition_lca.text_probs).

Trained on remarks with at least one tiered condition phrase whose latent class is confident
(probability > 0.9), excluding the ids in fallback_excluded_ids.csv.

  python tools/train_fallback.py listings.csv remarks_tags.csv condition_lca.csv \
         tools/fallback_excluded_ids.csv pipeline/assets/condition_fallback.json
"""
import json
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    listings, tags, lca, excl, out = sys.argv[1:6]
    r = pd.read_csv(listings, usecols=["ml_num", "remarks"])
    t = pd.read_csv(tags, usecols=["ml_num", "cond_n_atoms"])
    c = pd.read_csv(lca, usecols=["ml_num", "cond_lca_class", "cond_lca_conf"])
    ex = set(pd.read_csv(excl).ml_num)
    d = r.merge(t).merge(c)
    d = d[(d.cond_n_atoms > 0) & (d.cond_lca_conf > 0.9) & ~d.ml_num.isin(ex)]
    txt = d.remarks.fillna("").str.lower().str.replace("s/s", "stainless steel", regex=False)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=10, max_features=60000, sublinear_tf=True)
    X = vec.fit_transform(list(txt))
    clf = LogisticRegression(C=2.0, max_iter=2000)
    clf.fit(X, d.cond_lca_class.values)
    vocab = vec.vocabulary_
    terms = sorted(vocab, key=vocab.get)
    json.dump({"variant": "full", "classes": clf.classes_.tolist(), "terms": terms,
               "idf": vec.idf_.round(5).tolist(), "coef": clf.coef_.round(5).tolist(),
               "intercept": clf.intercept_.round(5).tolist()}, open(out, "w"))
    print(f"trained on {X.shape[0]} remarks, {X.shape[1]} terms -> {out}")
