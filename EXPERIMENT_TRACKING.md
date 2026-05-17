# Experiment Tracking Log

Use this document to systematically log all modeling attempts, feature engineering iterations, and leaderboard submissions. Maintaining this log ensures absolute reproducibility and tracking of performance improvements.

## 1. Experiment Log Ledger

| Date | Version | Model Type | Features Added / Modified | CV Score (WRMSSE) | Public LB Score | Notes & Next Steps |
|:---|:---|:---|:---|:---|:---|:---|
| 2026-05-18 | v0.1.0 | Historical Mean (28d) | Baseline: Daily Quantity Mean (last 28 days) | *TBD* | *TBD* | First baseline pipeline. Evaluation predictions set to 0. |
|            |        |            |                           |                   |                 |                                                    |
|            |        |            |                           |                   |                 |                                                    |
|            |        |            |                           |                   |                 |                                                    |
|            |        |            |                           |                   |                 |                                                    |

---

## 2. Guidelines for Rigorous Tracking
1. **Version Control Tagging**: Tag model versions in git matching the version listed in the tracking ledger (e.g., `git tag -a v0.1.0 -m "Baseline mean model"`).
2. **Local Validation Scheme**: Ensure that the local Cross-Validation (CV) setup mirrors the leaderboard's scoring rules. Use a rolling window validation (e.g., validation on the last 28 days of history) to prevent data leakage.
3. **Reproducibility**: Save all model seeds and configuration hyperparameters alongside the submission files to easily reproduce any logged result.
4. **Feature Impact Tracking**: Record a detailed summary of feature updates in [FEATURE_DICTIONARY.md](file:///c:/Users/Admin/OneDrive%20-%20National%20Economics%20University/Desktop/HBAAC/FEATURE_DICTIONARY.md) and link it in the ledger notes when features are added.
