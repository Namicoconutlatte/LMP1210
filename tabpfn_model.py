import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from tabpfn import TabPFNClassifier
from metrics import evaluate_model, compute_overall_metrics, find_threshold_youden

RANDOM_STATE = 42
TEST_SIZE = 0.2
AGE_MIN, AGE_MAX = 40, 70
TARGET = "HeartDisease"

NUMERIC_FEATURES = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
CATEGORICAL_FEATURES = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]


def load_and_preprocess():
    df = pd.read_excel("data/heart.xlsx")
    df = df[(df["Age"] >= AGE_MIN) & (df["Age"] <= AGE_MAX)].reset_index(drop=True)

    df.loc[df["Cholesterol"] == 0, "Cholesterol"] = pd.NA
    df.loc[df["RestingBP"] == 0, "RestingBP"] = pd.NA

    df["_strat"] = df["Sex"] + "_" + df[TARGET].astype(str)
    train, test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["_strat"]
    )
    train = train.drop(columns="_strat").reset_index(drop=True)
    test = test.drop(columns="_strat").reset_index(drop=True)

    for col in NUMERIC_FEATURES:
        median_val = train[col].median()
        train[col] = train[col].fillna(median_val)
        test[col] = test[col].fillna(median_val)

    for col in CATEGORICAL_FEATURES:
        mode_val = train[col].mode()[0]
        train[col] = train[col].fillna(mode_val)
        test[col] = test[col].fillna(mode_val)

    return train, test


def load_and_preprocess_sex_specific():
    df = pd.read_excel("data/heart.xlsx")
    df = df[(df["Age"] >= AGE_MIN) & (df["Age"] <= AGE_MAX)].reset_index(drop=True)
    df.loc[df["Cholesterol"] == 0, "Cholesterol"] = pd.NA
    df.loc[df["RestingBP"] == 0, "RestingBP"] = pd.NA

    splits = {}
    for sex in ["M", "F"]:
        df_sex = df[df["Sex"] == sex].reset_index(drop=True)
        train, test = train_test_split(
            df_sex, test_size=TEST_SIZE, random_state=RANDOM_STATE,
            stratify=df_sex[TARGET],
        )
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)

        # Impute
        for col in NUMERIC_FEATURES:
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)
            test[col] = test[col].fillna(median_val)
        for col in CATEGORICAL_FEATURES:
            mode_val = train[col].mode()[0]
            train[col] = train[col].fillna(mode_val)
            test[col] = test[col].fillna(mode_val)

        splits[sex] = (train, test)

    return splits


def main():
    train, test = load_and_preprocess()

    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET].values
    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET].values
    sex_labels = (test["Sex"] == "F").astype(int).values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = cross_val_predict(
        TabPFNClassifier(device="cuda:0", random_state=RANDOM_STATE),
        X_train, y_train, cv=cv, method="predict_proba",
    )[:, 1]
    threshold = find_threshold_youden(y_train, oof_proba)

    clf = TabPFNClassifier(device="cuda:0", random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_test)[:, 1]
    results = evaluate_model(y_test, y_proba, sex_labels, threshold)

    os.makedirs("predictions", exist_ok=True)
    pd.DataFrame({
        "y_true": y_test,
        "y_proba": y_proba,
        "sex_label": sex_labels,
    }).to_csv("predictions/tabpfn_predictions.csv", index=False)

    results["threshold"] = threshold
    with open("predictions/tabpfn_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved: predictions/tabpfn_predictions.csv, predictions/tabpfn_results.json")


    # Sex-specific TabPFN models
    splits = load_and_preprocess_sex_specific()
    comparison = {}

    for sex, label in [("M", "male"), ("F", "female")]:
        sex_train, sex_test = splits[sex]

        X_tr = sex_train.drop(columns=[TARGET])
        y_tr = sex_train[TARGET].values
        X_te = sex_test.drop(columns=[TARGET])
        y_te = sex_test[TARGET].values

        # Threshold selection for sex-specific model via Youden's J
        sex_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        oof_sex = cross_val_predict(
            TabPFNClassifier(device="cuda:0", random_state=RANDOM_STATE),
            X_tr, y_tr, cv=sex_cv, method="predict_proba",
        )[:, 1]
        sex_threshold = find_threshold_youden(y_tr, oof_sex)

        # Train sex-specific model
        clf_sex = TabPFNClassifier(device="cuda:0", random_state=RANDOM_STATE)
        clf_sex.fit(X_tr, y_tr)
        y_proba_sex = clf_sex.predict_proba(X_te)[:, 1]

        # Also evaluate global model on the same sex-specific test set
        y_proba_global = clf.predict_proba(X_te)[:, 1]

        sex_specific_metrics = compute_overall_metrics(y_te, y_proba_sex, sex_threshold)
        global_on_sex_metrics = compute_overall_metrics(y_te, y_proba_global, threshold)

        pd.DataFrame({
            "y_true": y_te,
            "y_proba_sex_specific": y_proba_sex,
            "y_proba_global": y_proba_global,
        }).to_csv(f"predictions/tabpfn_{label}_predictions.csv", index=False)

        sex_results = {
            "sex_specific": sex_specific_metrics,
            "sex_specific_threshold": sex_threshold,
            "global_on_subset": global_on_sex_metrics,
            "global_threshold": threshold,
        }
        with open(f"predictions/tabpfn_{label}_results.json", "w") as f:
            json.dump(sex_results, f, indent=2)

        comparison[label] = sex_results

    # Save combined comparison
    with open("predictions/tabpfn_sex_specific_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("\nSaved: predictions/tabpfn_{male,female}_predictions.csv")
    print("Saved: predictions/tabpfn_{male,female}_results.json")
    print("Saved: predictions/tabpfn_sex_specific_comparison.json")


if __name__ == "__main__":
    main()
