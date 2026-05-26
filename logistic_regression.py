import os
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from metrics import evaluate_model, find_threshold_youden

RANDOM_STATE = 42
NUMERIC_FEATURES = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
TARGET = "HeartDisease"
C_GRID = [0.01, 0.1, 1, 10, 100]


if __name__ == "__main__":
    train = pd.read_csv("data/global_train.csv")
    test = pd.read_csv("data/global_test.csv")

    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET].values
    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET].values
    sex_labels = test["Sex_F"].astype(int).values

    scaler = StandardScaler()
    X_train[NUMERIC_FEATURES] = scaler.fit_transform(X_train[NUMERIC_FEATURES])
    X_test[NUMERIC_FEATURES] = scaler.transform(X_test[NUMERIC_FEATURES])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        LogisticRegression(l1_ratio=0, solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE),
        param_grid={"C": C_GRID},
        scoring="roc_auc",
        cv=cv,
        refit=True,
    )
    grid.fit(X_train, y_train)

    best_c = grid.best_params_["C"]
    best_cv_auc = grid.best_score_

    oof_proba = cross_val_predict(
        grid.best_estimator_, X_train, y_train, cv=cv, method="predict_proba",
    )[:, 1]
    threshold = find_threshold_youden(y_train, oof_proba)
    y_proba = grid.predict_proba(X_test)[:, 1]
    results = evaluate_model(y_test, y_proba, sex_labels, threshold)

    os.makedirs("predictions", exist_ok=True)
    pd.DataFrame({
        "y_true": y_test,
        "y_proba": y_proba,
        "sex_label": sex_labels,
    }).to_csv("predictions/lr_predictions.csv", index=False)

    results["best_C"] = best_c
    results["cv_roc_auc"] = best_cv_auc
    results["threshold"] = threshold
    with open("predictions/lr_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved: predictions/lr_predictions.csv, predictions/lr_results.json")

