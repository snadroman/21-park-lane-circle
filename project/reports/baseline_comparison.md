# Our model vs the template baselines

Same walk-forward validation (2025-12 to 2026-05).

| Track | Model | MAPE | MdAPE | Median signed error |
|---|---|---|---|---|
| off_market | template baseline | 27.43% (26.95 to 27.94) | 20.84% | +8.07% |
| off_market | our model | 8.98% (8.80 to 9.17) | 6.69% | -0.22% |
| list_aware | template baseline | 8.09% (7.98 to 8.20) | 7.26% | +5.68% |
| list_aware | our model | 4.77% (4.69 to 4.85) | 3.73% | +0.00% |

Baselines: off_market = median price of the 10 nearest earlier sales of the same type;
list_aware = the list price itself.
