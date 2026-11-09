# Our pipeline

`python run.py predict` writes `submissions/submission_off_market.csv` and
`submissions/submission_list_aware.csv`. Everything is rebuilt from the training data on
each run: nothing is precomputed from sale prices.

## How it works (`models.py` → `pipeline/`)

1. **Remarks** (`pipeline/remarks.py`)
   - A phrase dictionary (`tag_remarks.py`) reads each listing's remarks.
   - A latent-class model (`condition_lca.py`) turns the condition phrases into a 1–5 condition score.
   - Remarks with fewer than two condition phrases use a TF-IDF + logistic-regression text model instead.
   - The dictionary also flags seller motivation.
   - Neither model saw a price. Their parameters are in `pipeline/assets/`.
2. **Market indexes** (`pipeline/market.py`)
   - Monthly price and sold/list indexes for four segments:
     - condo apartments (with co-ops)
     - detached homes
     - semis and freehold townhouses
     - condo townhouses
   - Built with a hedonic regression on the training sales, so each month's level is for a fixed typical home.
3. **Market forecast** for the four months after the training data:
   - Price: the average of two seasonal models (seasonal naive with drift, damped Holt-Winters).
   - Sold/list: the average of those seasonal models, a Temporal Fusion Transformer (`tft_forecast.py`) and last month's value.
4. **Per-home model** (`pipeline/homes.py`)
   - LightGBM predicts how a home prices *relative to* the market index at its **sale** month.
   - The sale month of a home to predict is unknown, so the prediction is averaged over the possible
     sale months (June–September for the test), weighted by how often homes sell that many months after listing.
   - Off-market: a linear hedonic fit per segment is LightGBM's starting point, so size and the other
     controls extend sensibly to the largest and smallest homes. Huber loss (±8%) focuses on the bulk of homes.
   - List-aware: one extra feature, how far the list price is above the home's estimated off-market value
     (for training homes, from models fitted only on earlier sales). Listings priced above their value sell further below asking.
   - Features:
     - structured columns
     - condition and motivation from the remarks
     - the market change between listing and sale month
     - the community's recent trend
     - earlier sales of the same condo building or house
     - nearby sales within about 300 m
     - list-aware only: the list price, raw, relative to the market and relative to the estimated value
   - Every history feature (community trend, building and house history, nearby sales) uses only sales dated
     strictly before the home's own sale.

## Results (walk-forward validation, Dec 2025 – May 2026, about 10,000 homes per track)

| Track | Model | MdAPE | MAPE | Median signed error |
|---|---|---|---|---|
| off_market | template baseline | 20.84% | 27.43% | +8.07% |
| off_market | **our model** | **6.69%** | **8.98%** (8.80–9.17) | −0.22% |
| list_aware | template baseline | 7.26% | 8.09% | +5.68% |
| list_aware | **our model** | **3.73%** | **4.77%** (4.69–4.85) | 0.00% |

Full report: `reports/validation_report.md`. Baseline comparison: `python compare_baselines.py`
(`reports/baseline_comparison.md`).

## Rebuilding the condition assets

The two files in `pipeline/assets/` are fitted once, from remark text only. To rebuild them
(needs scikit-learn for the last step):

```
python pipeline/tag_remarks.py listings.csv remarks_tags.csv
python pipeline/condition_lca.py fit remarks_tags.csv pipeline/assets/condition_lca_params.json
python pipeline/condition_lca.py apply remarks_tags.csv pipeline/assets/condition_lca_params.json condition_lca.csv
python tools/train_fallback.py listings.csv remarks_tags.csv condition_lca.csv \
       tools/fallback_excluded_ids.csv pipeline/assets/condition_fallback.json
```

## Reproducibility
- **Seeds:**
  - Everything random is seeded: LightGBM (`seed=20261028`, `deterministic=True`) and the TFT (seeds 0, 1, 2, Lightning deterministic mode, CPU).
  - PyTorch, NumPy and Python's global seeds are set at the start of every fit.
- **Threads:** fixed at 2, so results don't depend on how many cores the machine has.
- **Package versions:** pinned in `requirements.txt`.
- **Tested:** with those exact versions, two full `predict` runs gave byte-identical submission files.
- **Caveat:** other package versions (e.g. pandas 3) give slightly different numbers. A different CPU model could, in principle, change the last digits of some predictions.

## Timings (2 CPU cores, no GPU)
| Command | Time |
|---|---|
| `predict` | about 5 minutes (about 90 s of it is the TFT) |
| `validate` (6 monthly refits) | about 30 minutes |

**The TFT is required** (for the sold/list forecast). If darts, torch or pytorch-lightning can't be imported, the run stops
with an error and install instructions.
