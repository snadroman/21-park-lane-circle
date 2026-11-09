"""Monthly market indexes per property segment and their forecast.

Index: hedonic time-dummy regression per segment (condo apt incl. co-op / detached /
semi-detached + freehold townhouse / condo townhouse) on the training sales, by month of
sale. Controls: community, style, condition, size, rooms, parking, garage, lot and
motivation, so the index follows the market for a fixed typical home, not the mix sold.
    price      100 x exp(month level), 2021 average = 100
    sold_list  month level = typical log(sold / list)

Forecast for the 4 months after the last training month:
    price      seasonal combo
    sold_list  mean of (last month, seasonal combo, TFT)
"seasonal combo" = mean of seasonal-naive-with-drift and damped Holt-Winters. The TFT is
required: if darts / torch / pytorch-lightning cannot be imported, the run stops with an
error and install instructions.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

import forecast_classical as fc

SEGMENTS = {"condo_apt": ["Condo Apt", "Co-Op Apt"], "detached": ["Detached"],
            "semi_freehold": ["Semi-Detached", "Freehold Townhouse"], "condo_townhouse": ["Condo Townhouse"]}
SEGMENT_OF = {t: s for s, ts in SEGMENTS.items() for t in ts}

TARGETS = ("price", "sold_list")
H = fc.H
TFT_HELP = """
The market forecast needs the Temporal Fusion Transformer (darts + torch + pytorch-lightning),
and it could not be loaded, so the pipeline stops.

Install the pinned versions with:   pip install -r requirements.txt

The error was: {error}
"""


def _design(D, months):
    n = len(D)
    blocks = []
    mi = pd.Categorical(D.sold_m, categories=months).codes
    blocks.append(sp.csr_matrix((np.ones(n), (np.arange(n), mi)), shape=(n, len(months))))
    for col in ("community", "house_style", "cond_class"):
        cats = pd.Categorical(D[col].astype(str))
        codes = cats.codes
        keep = codes > 0
        blocks.append(sp.csr_matrix((np.ones(keep.sum()), (np.arange(n)[keep], codes[keep] - 1)),
                                    shape=(n, max(len(cats.categories) - 1, 1))))
    num = pd.DataFrame({
        "log_area": np.log(D.area_num.fillna(D.area_num.median())),
        "area_missing": D.area_num.isna().astype(float), "area_range": D.area_is_range.astype(float),
        "bedrooms": D.bedrooms.clip(0, 6), "washrooms": D.washrooms.clip(0, 6),
        "parking": D.parking_total.clip(0, 6), "garage": D.garage.fillna(0).clip(0, 4),
        "log_lot_front": np.log1p(D.lot_front.fillna(0)), "log_lot_depth": np.log1p(D.lot_depth.fillna(0)),
        "lot_missing": D.lot_front.isna().astype(float), "motivated": D.mot_any.astype(float)})
    num = num.loc[:, num.std() > 0]
    blocks.append(sp.csr_matrix(num.values))
    return sp.hstack(blocks).tocsr()


def build_indexes(T):
    """T: prepared training sales (see homes.prepare). Returns long table segment/target/month/value."""
    T = T[T.sold_m >= pd.Period("2021-01", "M")]
    months = pd.period_range(T.sold_m.min(), T.sold_m.max(), freq="M")
    rows = []
    for seg in SEGMENTS:
        D = T[T.segment == seg]
        X = _design(D, months)
        counts = D.sold_m.value_counts().reindex(months).fillna(0)
        for tgt in TARGETS:
            y = np.log(D.sold_price.values) if tgt == "price" else np.log(D.sold_price.values / D.list_price.values)
            beta = lsqr(X, y, atol=1e-10, btol=1e-10, iter_lim=20000)[0]
            ctrl = np.asarray(X[:, len(months):] @ beta[len(months):]).mean()
            eff = pd.Series(beta[:len(months)] + ctrl, index=months)
            eff[counts < 20] = np.nan
            eff = eff.interpolate(limit_direction="both")
            if tgt == "price":
                eff = 100 * np.exp(eff - eff[[m for m in months if m.year == 2021]].mean())
            rows.append(pd.DataFrame({"segment": seg, "target": tgt, "month": months.astype(str), "value": eff.values}))
    return pd.concat(rows, ignore_index=True)


def forecast(idx, origin):
    """Returns {(segment, target): Series by month}. Price series are log(index); months <= origin
    are the index itself, the next H months are the forecast."""
    origin = pd.Period(origin, "M")
    fut = pd.period_range(origin + 1, periods=H, freq="M")
    try:
        import tft_forecast
        tft = tft_forecast.fit_predict(idx, str(origin))      # one model over all 8 series
    except ImportError as error:
        raise RuntimeError(TFT_HELP.format(error=f"{type(error).__name__}: {error}")) from None
    tables = {}
    for (seg, tgt), g in idx.groupby(["segment", "target"]):
        y = fc.to_series(g).dropna()
        y = y[y.index <= origin]
        parts = [fc.f_combo(y, fut)]
        if tgt == "sold_list":
            t = tft[(tft.segment == seg) & (tft.target == tgt)].set_index("month").forecast
            parts += [t.reindex(fut.astype(str)).values, fc.f_naive(y, fut)]
        tables[(seg, tgt)] = pd.concat([y, pd.Series(np.mean(parts, axis=0), index=fut)])
    return tables


def lookup(tables, segment, months, target):
    """Index value for each (segment, month); months outside the range take the nearest end."""
    out = np.full(len(segment), np.nan)
    for seg in np.unique(segment):
        m = segment == seg
        s = tables[(seg, target)]
        ords = np.clip(pd.PeriodIndex(months[m], freq="M").asi8, s.index.asi8.min(), s.index.asi8.max())
        out[m] = s.reindex(pd.PeriodIndex.from_ordinals(ords, freq="M")).values
    return out
