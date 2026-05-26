## Methodological Notes

### 1. Classification Threshold Selection (Youden's J)

Classification thresholds were selected separately for each model using training-set cross-validation by maximizing Youden's J statistic.

**Youden's J:**

J = sensitivity + specificity - 1

For each model, we scanned all possible thresholds and selected the one maximizing J. This was done exclusively on training/validation data:

- **LR, XGBoost, TabPFN:** Thresholds were selected using out-of-fold predicted probabilities from 5-fold stratified cross-validation on the training set.
- **SCORE2:** Threshold was selected using SCORE2 probabilities computed on the training set.
- **Sex-specific models:** Thresholds were selected independently on each sex-specific training subset.

The selected thresholds were then fixed and applied to the held-out test set for all threshold-dependent metrics and fairness analyses. No test-set information was used in threshold selection.

**Rationale:** A fixed threshold of 0.5 assumes well-calibrated probabilities, which does not hold across models with different output scales (e.g., SCORE2 outputs 10-year CVD risk estimates, not binary classification probabilities). Model-specific threshold selection ensures each model is evaluated at its optimal operating point.

---

### 2. SCORE2 Probability Recalibration (Platt Scaling)

The raw SCORE2 approximation produced risk estimates on a scale poorly aligned with the binary HeartDisease outcome in this dataset (raw values ranged from ~0.003 to ~0.158, while disease prevalence was ~56%). This made the Brier score artificially high (~0.50) despite reasonable discriminative ability (AUC = 0.78).

To improve comparability and probability calibration, we applied logistic recalibration (Platt scaling) on the training set: a logistic regression model was fit using the raw SCORE2 risk as the sole predictor and HeartDisease as the outcome. The recalibrated probabilities were then applied to the held-out test set for evaluation.

This is a monotonic transformation that preserves the ranking of predictions (and therefore ROC-AUC) while rescaling the probabilities to better reflect the observed outcome frequency. After recalibration, the Brier score improved to ~0.23.

After this step, the baseline is best described as a **recalibrated SCORE2-inspired clinical baseline** rather than a pure published SCORE2 implementation.
