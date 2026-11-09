"""Classical forecasts of the monthly indexes, per series (log price index or sold/list).

    f_naive          last month's value, held flat
    f_snaive_drift   last value + last year's change over the same months
    holt_winters     damped additive Holt-Winters, 12-month season, parameters by grid search
    f_combo          mean of f_snaive_drift and holt_winters
"""
import itertools

import numpy as np
import pandas as pd

H = 4


def to_series(df):
    s = df.set_index(pd.PeriodIndex(df.month, freq="M")).value.astype(float)
    return np.log(s) if df.target.iloc[0] == "price" else s


def seasonal_factors(y):
    """Average deviation from a centred 12-month moving average, per calendar month."""
    ma = y.rolling(12, center=True, min_periods=12).mean().rolling(2, min_periods=2).mean().shift(-1)
    dev = (y - ma).dropna()
    s = dev.groupby(dev.index.month).mean().reindex(range(1, 13))
    if s.isna().any():                      # not enough history for some months
        s = s.fillna(0.0)
    return s - s.mean()


def f_naive(y, fut):
    return np.repeat(y.iloc[-1], len(fut))


def f_snaive_drift(y, fut):
    """Last value plus last year's change over the same months ("same as last year")."""
    out = []
    for m in fut:
        a, b = m - 12, y.index[-1] - 12
        out.append(y.iloc[-1] + (y.get(a, np.nan) - y.get(b, np.nan)))
    return np.array(out)


def f_seasonal_level(y, fut, k=3):
    s = seasonal_factors(y)
    des = y - s.reindex(y.index.month).values
    level = des.iloc[-k:].mean()
    return np.array([level + s[m.month] for m in fut])


def holt_winters(y, fut, grid=(0.1, 0.3, 0.5, 0.7), phis=(0.8, 0.9, 0.98)):
    """Additive damped Holt-Winters; parameters picked by in-sample one-step SSE."""
    v = y.values
    m = 12
    if len(v) < 2 * m:
        return f_seasonal_level(y, fut)
    best = None
    s0 = seasonal_factors(y).reindex(y.index.month[:m]).values
    for a, b, g, ph in itertools.product(grid, (0.05, 0.1, 0.2), grid, phis):
        lvl, tr, seas = v[:m].mean() - 0.0, 0.0, list(s0)
        sse = 0.0
        for t in range(len(v)):
            st = seas[t]
            fc = lvl + ph * tr + st
            if t >= m:
                sse += (v[t] - fc) ** 2
            new_l = a * (v[t] - st) + (1 - a) * (lvl + ph * tr)
            tr = b * (new_l - lvl) + (1 - b) * ph * tr
            seas.append(g * (v[t] - new_l) + (1 - g) * st)
            lvl = new_l
        if best is None or sse < best[0]:
            best = (sse, lvl, tr, seas, ph)
    _, lvl, tr, seas, ph = best
    n = len(v)
    return np.array([lvl + tr * sum(ph ** i for i in range(1, h + 1)) + seas[n - m + (h - 1) % m]
                     for h in range(1, len(fut) + 1)])


def f_combo(y, fut):
    """Average of seasonal naive + drift and Holt-Winters (forecast combination)."""
    return (f_snaive_drift(y, fut) + holt_winters(y, fut)) / 2
