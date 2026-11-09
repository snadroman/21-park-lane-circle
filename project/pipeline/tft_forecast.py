"""Temporal Fusion Transformer for the monthly index forecasts (darts + torch).

One global model is trained on all 8 series (4 segments x {price, sold_list}) up to the
forecast origin. Static covariates tell it which series is which; month-of-year and a
relative time index are known future inputs. Point forecasts (MSE loss), averaged over a
fixed set of seeds. Training runs on CPU with deterministic settings so results reproduce.
market.forecast uses its sold/list forecasts.
"""
import numpy as np
import pandas as pd

H = 4
SEEDS = (0, 1, 2)
PARAMS = dict(input_chunk_length=12, output_chunk_length=H, hidden_size=16, lstm_layers=1,
              num_attention_heads=2, dropout=0.1, batch_size=16, n_epochs=80)


def _series(df, segments):
    """darts TimeSeries for one segment/target (log for price), with static covariates
    (one-hot property type + price/sold-list flag)."""
    from darts import TimeSeries
    seg, tgt = df.segment.iloc[0], df.target.iloc[0]
    d = df.dropna(subset=["value"]).copy()
    d["time"] = pd.PeriodIndex(d.month, freq="M").to_timestamp()
    d["y"] = np.log(d.value) if tgt == "price" else d.value
    ts = TimeSeries.from_dataframe(d, time_col="time", value_cols="y", freq="MS")
    static = pd.DataFrame({**{f"seg_{x}": [float(seg == x)] for x in segments[1:]},
                           "is_price": [float(tgt == "price")]})
    return ts.with_static_covariates(static), seg, tgt


def fit_predict(idx_df, origin, seeds=SEEDS, accelerator="cpu"):
    """Train on the indexes up to `origin` and forecast origin+1..origin+H for all 8 series."""
    import torch
    from darts.dataprocessing.transformers import Scaler
    # Import from the module directly: `from darts.models import TFTModel` goes through a
    # catch-all loader that marks every torch model unavailable if any one of them is
    # missing an optional dependency.
    from darts.models.forecasting.tft_model import TFTModel

    series, keys = [], []
    segments = sorted(idx_df.segment.unique())
    for (seg, tgt), g in idx_df.groupby(["segment", "target"]):
        g = g[pd.PeriodIndex(g.month, freq="M") <= pd.Period(origin, "M")]
        ts, s, t = _series(g, segments)
        series.append(ts)
        keys.append((s, t))
    scaler = Scaler()
    scaled = scaler.fit_transform(series)
    preds = []
    for seed in seeds:
        model = TFTModel(**PARAMS, add_relative_index=True,
                         add_encoders={"cyclic": {"future": ["month"]}},
                         likelihood=None, loss_fn=torch.nn.MSELoss(), random_state=seed,
                         pl_trainer_kwargs={"accelerator": accelerator, "deterministic": True,
                                            "enable_progress_bar": False, "enable_model_summary": False,
                                            "logger": False, "enable_checkpointing": False})
        model.fit(scaled, verbose=False)
        p = scaler.inverse_transform(model.predict(n=H, series=scaled, verbose=False))
        preds.append(np.stack([np.asarray(x.values()).ravel() for x in p]))
    mean = np.mean(preds, axis=0)
    fut = pd.period_range(pd.Period(origin, "M") + 1, periods=H, freq="M")
    rows = []
    for (seg, tgt), f in zip(keys, mean):
        for h, (m, v) in enumerate(zip(fut, f), 1):
            rows.append(dict(origin=origin, segment=seg, target=tgt, h=h, month=str(m), forecast=v))
    return pd.DataFrame(rows)
