import os
import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from metrics import evaluate_model, find_threshold_youden

# SCORE2 beta definition
# beta_1=c_age, beta_2=smoking, beta_3=c_sbp, beta_4=diabetes, beta_5=c_tchol, beta_6=c_hdl,
# beta_7=c_age*smoking, beta_8=c_age*c_sbp, beta_9=c_age*c_tchol, beta_10=c_age*c_hdl,
# beta_11=c_age*diabetes

MALE_BETAS = np.array([
    0.3742, 0.6012, 0.2777, 0.6457, 0.1458, -0.2698,
    -0.0755, -0.0255, -0.0281, 0.0426, -0.0983,
])

FEMALE_BETAS = np.array([
    0.4648, 0.7744, 0.3131, 0.8096, 0.1002, -0.2606,
    -0.1088, -0.0277, -0.0226, 0.0613, -0.1272,
])

S0_MALE = 0.9605
S0_FEMALE = 0.9776

SMOKING = 0
HDL = 1.3  # mmol/L
CHOL_MGDL_TO_MMOL = 38.67


def compute_score2_risk(df):

    n = len(df)

    age = df["Age"].values.astype(float)
    sbp = df["RestingBP"].values.astype(float)
    tchol = df["Cholesterol"].values.astype(float) / CHOL_MGDL_TO_MMOL
    diabetes = df["FastingBS"].values.astype(float)
    is_female = df["Sex_F"].values.astype(bool)

    # Transformed variables
    c_age = (age - 60) / 5
    c_sbp = (sbp - 120) / 20
    c_tchol = tchol - 6
    c_hdl = (HDL - 1.3) / 0.5  # = 0.0
    smoking = np.full(n, SMOKING, dtype=float)

    X = np.column_stack([
        c_age,
        smoking,
        c_sbp,
        diabetes,
        c_tchol,
        np.full(n, c_hdl),
        c_age * smoking,
        c_age * c_sbp,
        c_age * c_tchol,
        c_age * c_hdl,
        c_age * diabetes,
    ])

    # Sex-specific linear predictor
    male_mask = ~is_female
    female_mask = is_female

    lp = np.zeros(n)
    lp[male_mask] = X[male_mask] @ MALE_BETAS
    lp[female_mask] = X[female_mask] @ FEMALE_BETAS

    # Sex-specific uncalibrated risk
    risk = np.zeros(n)
    risk[male_mask] = 1 - S0_MALE ** np.exp(lp[male_mask])
    risk[female_mask] = 1 - S0_FEMALE ** np.exp(lp[female_mask])

    return np.clip(risk, 0.0, 1.0)


def main():
    train = pd.read_csv("data/global_train.csv")
    test = pd.read_csv("data/global_test.csv")

    y_train = train["HeartDisease"].values
    y_test = test["HeartDisease"].values
    sex_labels = test["Sex_F"].astype(int).values

    # Compute raw SCORE2 risk
    train_risk_raw = compute_score2_risk(train)
    test_risk_raw = compute_score2_risk(test)

    # Platt scaling (logistic recalibration) on training set
    platt = LogisticRegression(solver="lbfgs", max_iter=1000)
    platt.fit(train_risk_raw.reshape(-1, 1), y_train)
    train_risk = platt.predict_proba(train_risk_raw.reshape(-1, 1))[:, 1]
    test_risk = platt.predict_proba(test_risk_raw.reshape(-1, 1))[:, 1]

    threshold = find_threshold_youden(y_train, train_risk)
    results = evaluate_model(y_test, test_risk, sex_labels, threshold)

    os.makedirs("predictions", exist_ok=True)
    pd.DataFrame({
        "y_true": y_test,
        "y_proba": test_risk,
        "sex_label": sex_labels,
    }).to_csv("predictions/score2_predictions.csv", index=False)

    results["threshold"] = threshold
    with open("predictions/score2_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved: predictions/score2_predictions.csv, predictions/score2_results.json")


if __name__ == "__main__":
    main()
