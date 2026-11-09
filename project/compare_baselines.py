"""Score the template's two baselines with the same walk-forward validation as run.py,
for comparison with our model (reports/validation_report.md).

    python compare_baselines.py      writes reports/baseline_comparison.md
"""
from pathlib import Path

import numpy as np
import pandas as pd

import data
import models
from run import TRACKS, VALIDATION_MONTHS, metrics

ROOT = Path(__file__).parent


def main():
    listings = data.load("listings")
    rows = []
    for month in VALIDATION_MONTHS:
        train = listings[listings["date_sold"] < month.start_time]
        homes = listings[listings["date_sold"].dt.to_period("M") == month]
        for track in TRACKS:
            model = models.ListPriceBaseline() if track == "list_aware" else models.NearestSalesBaseline(k=10)
            predicted = model.fit(train).predict(data.hide_answers(homes, track))
            rows.append(pd.DataFrame({"track": track, "sold_price": homes["sold_price"].values,
                                      "predicted_price": np.round(predicted)}))
        print(month, "done")
    base = pd.concat(rows)
    ours = pd.read_csv(ROOT / "outputs" / "validation_predictions.csv")
    lines = ["# Our model vs the template baselines", "",
             f"Same walk-forward validation ({VALIDATION_MONTHS[0]} to {VALIDATION_MONTHS[-1]}).", "",
             "| Track | Model | MAPE | MdAPE | Median signed error |", "|---|---|---|---|---|"]
    for track in TRACKS:
        for name, df in (("template baseline", base), ("our model", ours)):
            s = metrics(df[df["track"] == track])
            lines.append(f"| {track} | {name} | {s['mape']:.2f}% ({s['mape_95ci'][0]:.2f} to {s['mape_95ci'][1]:.2f}) "
                         f"| {s['mdape']:.2f}% | {s['median_signed_error']:+.2f}% |")
    lines += ["", "Baselines: off_market = median price of the 10 nearest earlier sales of the same type;",
              "list_aware = the list price itself."]
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "baseline_comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
