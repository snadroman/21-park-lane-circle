"""Per-home models on top of the market indexes.

Targets are relative to the market at the sale month (prices in this data follow the market when a home sells):
  off_market:  log(sold) - log(price index[segment, sale month])
  list_aware:  log(sold / list) - sold/list index[segment, sale month]
The sale month of a home to predict is unknown, so the prediction averages log prices over the possible sale
months (the month after the training data up to the last listing month among the homes), weighted by how
often homes sell that many months after listing:
  off_market:  (forecast) price index[s] x exp(hedonic base + LightGBM)
  list_aware:  list price x sold/list index[s] x exp(LightGBM)

off_market: a per-segment linear hedonic fit is LightGBM's starting point (init_score), so size and the other
  controls extend sensibly to the largest and smallest homes; Huber loss (+-8%) targets the bulk of homes.
list_aware: 'over' = log list price minus the home's estimated off-market value (for training rows, from models
  fitted only on earlier sales): listings priced above their value sell further below asking.

Features (no raw latitude / longitude):
  community, property type, style, rooms, parking, garage, size, lot
  cond_exp, cond_class                 condition from the remarks (remarks.py)
  mot_*                                seller motivation from the remarks
  drift                                index change from the listing month to the (candidate) sale month
  comm_trend_12m, comm_n_12m           community+segment: recent 12 months vs usual level
  bld_level, bld_n                     earlier sales at the same exact point (condo building /
                                       townhouse complex / the same house), relative to the index
  prev_rel, prev_age                   the same house's previous sale (same point, style, bedrooms)
  loc_level, loc_weight, loc_level_same  nearby sales (Gaussian weights, sigma 150 m, cut at 450 m) at other
                                       points; all communities / own community
  log_list, log_list_rel, over         list_aware only: the list price, relative to the price index at
                                       listing, and relative to the home's estimated value

Every history feature uses only sales dated strictly before the home's own sale: up to the month before
(community, building, previous sale) or the day before (nearby sales) for a training sale, and all training
sales for a home to predict.
"""
import numpy as np
import pandas as pd

from . import SEED, THREADS, market
from .remarks import MOTIVATION, remark_features

CATS = ["house_type_name", "house_style", "community", "segment"]
STRUCT = ["bedrooms", "washrooms", "parking_total", "garage", "area_num", "area_is_range", "lot_front", "lot_depth"]
COND = ["cond_exp", "cond_class"]
HIST = ["comm_trend_12m", "comm_n_12m", "bld_level", "bld_n", "prev_rel", "prev_age",
        "loc_level", "loc_weight", "loc_level_same"]
MODEL_FEATURES = {
    "off_market": CATS + STRUCT + COND + ["mot_any"] + HIST + ["drift"],
    "list_aware": CATS + STRUCT + COND + MOTIVATION + HIST + ["log_list", "log_list_rel", "drift", "over"],
}
LGB = dict(objective="regression_l1", learning_rate=0.03, num_leaves=63, min_child_samples=40,
           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
           n_estimators=4000, seed=SEED, deterministic=True, force_row_wise=True, num_threads=THREADS,
           verbose=-1)
OFF_LGB = dict(objective="huber", alpha=0.08)   # off-market: squared loss inside +-8%, absolute beyond
STACK_LGB = dict(n_estimators=1000, learning_rate=0.06)   # the smaller value model behind 'over'


# ------------------------------------------------------------------------- basic columns
def prepare(df):
    D = df.copy()
    D = D.join(remark_features(D))
    D["segment"] = D.house_type_name.map(market.SEGMENT_OF).fillna("condo_apt")
    D["list_m"] = pd.to_datetime(D.date_listed).dt.to_period("M")
    D["list_m"] = D.list_m.where(D.list_m >= pd.Period("2021-01", "M"), pd.Period("2021-01", "M"))
    if "date_sold" in D:
        D["sold_m"] = pd.to_datetime(D.date_sold).dt.to_period("M")
    a = D.house_area.astype(str)
    r = a.str.extract(r"^(\d+)\s*-\s*(\d+)$").astype(float)
    D["area_num"] = np.where(r[0].notna(), (r[0] + r[1]) / 2, pd.to_numeric(a, errors="coerce"))
    D["area_is_range"] = r[0].notna().astype(int)
    D["pt"] = D.latitude.round(6).astype(str) + "," + D.longitude.round(6).astype(str)
    if "list_price" in D:
        D["log_list"] = np.log(D.list_price)
    return D


def _known(D, origin):
    """Last month of sales a row's history features may use: the month before its sale for a training
    sale, the last training month for a home to predict."""
    if "sold_m" in D:
        return pd.PeriodIndex([min(m - 1, origin) for m in D.sold_m], freq="M")
    return pd.PeriodIndex([origin] * len(D), freq="M")


# --------------------------------------------------------------------- history features
def _xy(D):
    return np.c_[(D.longitude.values + 79.38) * np.cos(np.radians(43.65)) * 111320,
                 (D.latitude.values - 43.65) * 110540]


def add_history(T, Q, tab, origin):
    """Add trend, building, previous-sale and local-field features to Q, from training sales T."""
    from scipy.spatial import cKDTree
    Q = Q.copy()
    known = _known(Q, origin)
    ix = market.lookup(tab, T.segment.astype(str).values, T.sold_m.values, "price")
    rel_sqft = np.log(T.sold_price.values / T.area_num.values) - ix
    rel_price = np.log(T.sold_price.values) - ix
    tkey = (T.community.astype(str) + "|" + T.segment.astype(str)).values
    qkey = (Q.community.astype(str) + "|" + Q.segment.astype(str)).values

    # community + segment: recent 12 months vs usual level (shrunk), as of each row's known month
    months = pd.period_range("2021-01", origin, freq="M")
    S0 = pd.DataFrame({"c": tkey, "m": T.sold_m.values, "v": rel_sqft})
    S0 = S0[np.isfinite(S0.v) & (S0.m >= months[0])]
    S = S0.pivot_table(index="m", columns="c", values="v", aggfunc="sum").reindex(months).fillna(0)
    N = S0.pivot_table(index="m", columns="c", values="v", aggfunc="count").reindex(months).fillna(0)
    # recent is shrunk toward the community's own usual level (not toward 0), so a community
    # that is steadily above / below its segment shows no trend
    usual = S.cumsum() / (N.cumsum() + 20)
    recent = (S.rolling(12, min_periods=1).sum() + 20 * usual) / (N.rolling(12, min_periods=1).sum() + 20)
    n12 = N.rolling(12, min_periods=1).sum()
    ci = pd.Index(S.columns).get_indexer(qkey)
    ri = np.clip(known.asi8, months.asi8[0], months.asi8[-1]) - months.asi8[0]
    ok = (ci >= 0) & (ri >= 0)
    trend, cnt = np.full(len(Q), np.nan), np.zeros(len(Q))
    trend[ok] = (recent.values - usual.values)[ri[ok], ci[ok]]
    cnt[ok] = n12.values[ri[ok], ci[ok]]
    Q["comm_trend_12m"], Q["comm_n_12m"] = trend, cnt

    # same exact point (building / complex / house): earlier sales up to each row's known month
    H = pd.DataFrame({"pt": T.pt.values, "key2": (T.pt + "|" + T.house_style.astype(str) + "|" + T.bedrooms.astype(str)).values,
                      "t": T.sold_m.dt.to_timestamp(how="end").values.astype("datetime64[ns]"), "rs": rel_sqft, "rp": rel_price})
    q = pd.DataFrame({"_i": np.arange(len(Q)), "pt": Q.pt.values,
                      "key2": (Q.pt + "|" + Q.house_style.astype(str) + "|" + Q.bedrooms.astype(str)).values,
                      "t": known.to_timestamp(how="end").values.astype("datetime64[ns]")})
    g = (H[np.isfinite(H.rs)].groupby(["pt", "t"]).rs.agg(["sum", "count"]).groupby(level=0).cumsum()
         .reset_index().sort_values("t"))
    m = pd.merge_asof(q.sort_values("t"), g, on="t", by="pt", direction="backward").sort_values("_i")
    Q["bld_n"] = m["count"].fillna(0).values
    Q["bld_level"] = (m["sum"].fillna(0) / (m["count"].fillna(0) + 5)).values
    # several earlier sales of the same house in one month: take the latest-dated one (stable, data-order free)
    r = (H[["key2", "t", "rp"]].assign(t_prev=lambda d: d.t, d=pd.to_datetime(T.date_sold).values)
         .sort_values(["t", "d", "rp"], kind="mergesort"))
    m2 = pd.merge_asof(q.sort_values("t"), r, on="t", by="key2", direction="backward").sort_values("_i")
    Q["prev_rel"] = m2.rp.values
    Q["prev_age"] = ((q.t - m2.t_prev.values).dt.days / 30.4).values

    # local price field: Gaussian-weighted nearby sales at other points (sigma 150 m, cut 450 m), using only
    # sales dated strictly before the home's own sale (training rows); homes to predict see all training sales.
    t_T = pd.to_datetime(T.date_sold).values.astype("datetime64[D]").astype(np.int64)
    t_Q = (pd.to_datetime(Q.date_sold).values.astype("datetime64[D]").astype(np.int64) if "date_sold" in Q
           else np.full(len(Q), np.iinfo(np.int64).max))
    # each neighbour's per-sqft level relative to its segment's average over sales strictly before it
    seg = T.segment.astype(str).values
    g = pd.DataFrame({"seg": seg, "t": t_T, "v": rel_sqft})
    g = g[np.isfinite(g.v)].groupby(["seg", "t"]).v.agg(["sum", "count"]).groupby(level=0).cumsum()
    g = g.groupby(level=0).shift(1)                       # totals over earlier dates only
    prior = (g["sum"] / g["count"]).reindex(pd.MultiIndex.from_arrays([seg, t_T])).values
    v = rel_sqft - prior
    okv = np.isfinite(v)
    Xh, vh, hp, hc, ht = _xy(T)[okv], v[okv], T.pt.values[okv], T.community.astype(str).values[okv], t_T[okv]
    tree = cKDTree(Xh)
    Xq, qp, qc = _xy(Q), Q.pt.values, Q.community.astype(str).values
    acc = np.zeros((len(Q), 4))
    for c0 in range(0, len(Q), 20000):
        n = min(20000, len(Q) - c0)
        M = cKDTree(Xq[c0:c0 + n]).sparse_distance_matrix(tree, 450.0, output_type="coo_matrix")
        rr, cc, dd = M.row, M.col, M.data
        keep = (hp[cc] != qp[c0 + rr]) & (ht[cc] < t_Q[c0 + rr])
        rr, cc, dd = rr[keep], cc[keep], dd[keep]
        w = np.exp(-0.5 * (dd / 150.0) ** 2)
        same = hc[cc] == qc[c0 + rr]
        acc[c0:c0 + n, 0] = np.bincount(rr, w * vh[cc], minlength=n)
        acc[c0:c0 + n, 1] = np.bincount(rr, w, minlength=n)
        acc[c0:c0 + n, 2] = np.bincount(rr, w * vh[cc] * same, minlength=n)
        acc[c0:c0 + n, 3] = np.bincount(rr, w * same, minlength=n)
    Q["loc_level"] = acc[:, 0] / (acc[:, 1] + 2)
    Q["loc_weight"] = acc[:, 1]
    Q["loc_level_same"] = acc[:, 2] / (acc[:, 3] + 2)
    return Q


# ------------------------------------------------------------------- shared training state
_context_cache = {}


def context(train, log=print):
    """Everything both tracks share for one training set: remarks, indexes, forecast,
    and the training rows' features. Cached, so the two tracks build it once."""
    key = (len(train), str(pd.to_datetime(train.date_sold).max()), int(pd.util.hash_pandas_object(train.ml_num, index=False).sum()))
    if key in _context_cache:
        return _context_cache[key]
    from . import make_deterministic
    make_deterministic()
    import time
    t0 = time.time()
    T = prepare(train[pd.to_datetime(train.date_sold) >= "2021-01-01"].reset_index(drop=True))
    origin = T.sold_m.max()
    t1 = time.time(); idx = market.build_indexes(T)
    t2 = time.time(); tab = market.forecast(idx, origin)
    t3 = time.time(); T = add_history(T, T, tab, origin)
    log(f"   timings: remarks+prepare {t1 - t0:.0f}s, indexes {t2 - t1:.0f}s, forecast {t3 - t2:.0f}s, "
        f"history features {time.time() - t3:.0f}s")
    T["log_list_rel"] = T.log_list.values - market.lookup(tab, T.segment.values, T.list_m.values, "price")
    cats = {c: pd.Categorical(T[c].astype(str)).categories for c in CATS}
    for c in CATS:
        T[c] = pd.Categorical(T[c].astype(str), categories=cats[c])
    ctx = dict(T=T, origin=origin, idx=idx, tab=tab, cats=cats)
    log(f"   market built to {origin}, {len(T):,} training sales")
    _context_cache.clear()
    _context_cache[key] = ctx
    return ctx


# ------------------------------------------------------------------- hedonic baseline
def _hedonic_design(D, levels, fill):
    """Linear hedonic design (no time terms): intercept, community / style / condition dummies (levels fixed
    from the training rows, first level dropped) and the numeric controls of the market index."""
    import scipy.sparse as sp
    n = len(D); blocks = [sp.csr_matrix(np.ones((n, 1)))]
    for col in ("community", "house_style", "cond_class"):
        codes = pd.Categorical(D[col].astype(str), categories=levels[col]).codes
        keep = codes > 0
        blocks.append(sp.csr_matrix((np.ones(keep.sum()), (np.arange(n)[keep], codes[keep] - 1)),
                                    shape=(n, max(len(levels[col]) - 1, 1))))
    num = np.c_[np.log(D.area_num.fillna(fill)), D.area_num.isna(), D.area_is_range,
                D.bedrooms.clip(0, 6), D.washrooms.clip(0, 6), D.parking_total.clip(0, 6), D.garage.fillna(0).clip(0, 4),
                np.log1p(D.lot_front.fillna(0)), np.log1p(D.lot_depth.fillna(0)), D.lot_front.isna(), D.mot_any]
    blocks.append(sp.csr_matrix(num.astype(float)))
    return sp.hstack(blocks).tocsr()


def hedonic_fit(T, y):
    """Per-segment linear fit of a relative-price target; LightGBM then learns what's left. A linear model in
    log size etc. extends sensibly to the largest and smallest homes, where trees flatten out."""
    from scipy.sparse.linalg import lsqr
    out = {}
    for seg in T.segment.astype(str).unique():
        m = (T.segment.astype(str) == seg).values; D = T[m]
        levels = {c: list(pd.Categorical(D[c].astype(str)).categories) for c in ("community", "house_style", "cond_class")}
        fill = float(D.area_num.median())
        b = lsqr(_hedonic_design(D, levels, fill), y[m], atol=1e-10, btol=1e-10, iter_lim=20000)[0]
        out[seg] = (levels, fill, b)
    return out


def hedonic_predict(H, D):
    base = np.zeros(len(D)); seg = D.segment.astype(str).values
    for s, (levels, fill, b) in H.items():
        m = seg == s
        if m.any(): base[m] = _hedonic_design(D[m], levels, fill) @ b
    return base


# ------------------------------------------------------------------- models
def _lgb(params, X, y, init=None):
    import lightgbm as lgb
    from . import make_deterministic
    make_deterministic()
    m = lgb.LGBMRegressor(**{**LGB, **params})
    m.fit(X, y, categorical_feature=[c for c in CATS if c in X.columns], init_score=init)
    return m


def _stack_value(T, y):
    """Off-market value estimate for the list-aware 'overpricing' feature: the off-market recipe (sale-month
    target y, hedonic base, Huber) as a smaller 1000-tree model, read at the listing month (drift = 0).
    Training rows: half-year blocks, each valued by a model fitted only on sales before the block starts
    (the first half-year has no earlier sales and is left missing). Homes to predict: a model on all
    training sales."""
    f, params = MODEL_FEATURES["off_market"], {**OFF_LGB, **STACK_LGB}
    T0 = T.assign(drift=0.0)
    sold = pd.to_datetime(T.date_sold)
    block = (sold.dt.year * 2 + (sold.dt.month > 6)).values
    val = np.full(len(T), np.nan)
    for b in np.unique(block)[1:]:
        tr, te = block < b, block == b
        H = hedonic_fit(T[tr], y[tr])
        m = _lgb(params, T.loc[tr, f], y[tr], hedonic_predict(H, T[tr]))
        val[te] = m.predict(T0.loc[te, f]) + hedonic_predict(H, T0[te])
    H = hedonic_fit(T, y); full = _lgb(params, T[f], y, hedonic_predict(H, T))

    def value(D):
        D0 = D.assign(drift=0.0)
        return full.predict(D0[f]) + hedonic_predict(H, D0)
    return val, value


def fit(ctx, track):
    """Fit one track. Targets are relative to the market index at the sale month; 'drift' = index change from
    the listing month to the sale month."""
    T = ctx["T"].copy()
    seg = T.segment.astype(str).values
    ip_s = market.lookup(ctx["tab"], seg, T.sold_m.values, "price")
    T["drift"] = ip_s - market.lookup(ctx["tab"], seg, T.list_m.values, "price")
    k = np.array([(s - l).n for s, l in zip(T.sold_m, T.list_m)]).clip(0, 24)
    model = {"feats": MODEL_FEATURES[track], "p_months": np.bincount(k, minlength=25) / len(k)}
    if track == "off_market":
        y = np.log(T.sold_price.values) - ip_s
        model["hedonic"] = hedonic_fit(T, y)
        model["lgb"] = _lgb(OFF_LGB, T[model["feats"]], y, hedonic_predict(model["hedonic"], T))
    else:
        oof, model["stack"] = _stack_value(T, np.log(T.sold_price.values) - ip_s)
        T["over"] = T.log_list.values - (market.lookup(ctx["tab"], seg, T.list_m.values, "price") + oof)
        y = np.log(T.sold_price.values / T.list_price.values) - market.lookup(ctx["tab"], seg, T.sold_m.values, "sold_list")
        model["lgb"] = _lgb({}, T[model["feats"]], y)
    return model


def predict(ctx, model, homes, track):
    Q = prepare(homes.reset_index(drop=True))
    Q = add_history(ctx["T"], Q, ctx["tab"], ctx["origin"])
    start = ctx["origin"] + 1; end = max(Q.list_m.max(), start)
    seg = Q.segment.astype(str).values
    ip_l = market.lookup(ctx["tab"], seg, Q.list_m.values, "price")
    if "log_list" in Q:
        Q["log_list_rel"] = Q.log_list.values - ip_l
    for c in CATS:
        Q[c] = pd.Categorical(Q[c].astype(str), categories=ctx["cats"][c])
    base = hedonic_predict(model["hedonic"], Q) if track == "off_market" else 0.0
    if track == "list_aware":
        Q["over"] = Q.log_list.values - (ip_l + model["stack"](Q))
    # The sale month is unknown: the homes sold some time between the month after the training data and the
    # last listing month among them (June-September for the test). Average log prices over those months,
    # weighted by how often homes sell that many months after listing.
    pk = model["p_months"]
    num, den = np.zeros(len(Q)), np.zeros(len(Q))
    for s in pd.period_range(start, end, freq="M"):
        off = np.array([(s - l).n for l in Q.list_m]); w = np.where(off >= 0, pk[off.clip(0, 24)], 0.0)
        if not w.any():
            continue
        sm = np.array([s] * len(Q)); ip_s = market.lookup(ctx["tab"], seg, sm, "price")
        Q["drift"] = ip_s - ip_l
        rel = model["lgb"].predict(Q[model["feats"]]) + base
        lp = ip_s + rel if track == "off_market" else Q.log_list.values + market.lookup(ctx["tab"], seg, sm, "sold_list") + rel
        num += w * lp; den += w
    return np.exp(num / np.where(den > 0, den, 1))
