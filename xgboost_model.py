import os
import pandas as pd
import json
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from metrics import evaluate_model, find_threshold_youden

RANDOM_STATE = 42
TARGET = "HeartDisease"
PARAM_GRID = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.3],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

if __name__ == "__main__":
    train = pd.read_csv("data/global_train.csv")
    test = pd.read_csv("data/global_test.csv")

    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET].values
    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET].values
    sex_labels = test["Sex_F"].astype(int).values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        XGBClassifier(use_label_encoder=False,eval_metric="logloss",random_state=RANDOM_STATE),
        param_grid=PARAM_GRID,
        scoring="roc_auc",
        cv=cv,
        refit=True,
        n_jobs=64,
        verbose=3,
    )
    grid.fit(X_train, y_train)

    best_params = grid.best_params_
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
    }).to_csv("predictions/xgboost_predictions.csv", index=False)

    results["best_params"] = best_params
    results["cv_roc_auc"] = best_cv_auc
    results["threshold"] = threshold
    with open("predictions/xgboost_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved: predictions/xgboost_predictions.csv, predictions/xgboost_results.json")

