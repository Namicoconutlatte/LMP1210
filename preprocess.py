import os
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42 # re-producible
TEST_SIZE = 0.2 # 80/20 split

AGE_MIN, AGE_MAX = 40, 70
TARGET = "HeartDisease"

NUMERIC_FEATURES = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
CATEGORICAL_FEATURES = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

OUTPUT_DIR = "data"

df = pd.read_excel("data/heart.xlsx")
df = df[(df["Age"] >= AGE_MIN) & (df["Age"] <= AGE_MAX)].reset_index(drop=True)

df.loc[df["Cholesterol"] == 0, "Cholesterol"] = pd.NA
df.loc[df["RestingBP"] == 0, "RestingBP"] = pd.NA

missing_rates = (df.isnull().sum() / len(df) * 100).round(2)

def split_data(data, stratify_col):

    train, test = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data[stratify_col],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


# Global split — stratify by Sex + HeartDisease
df["_strat"] = df["Sex"] + "_" + df[TARGET].astype(str)
global_train, global_test = split_data(df, "_strat")
global_train = global_train.drop(columns="_strat")
global_test = global_test.drop(columns="_strat")
df = df.drop(columns="_strat")

# Male subset
df_male = df[df["Sex"] == "M"].reset_index(drop=True)
male_train, male_test = split_data(df_male, TARGET)

# Female subset
df_female = df[df["Sex"] == "F"].reset_index(drop=True)
female_train, female_test = split_data(df_female, TARGET)


def impute(train, test, numeric_cols, categorical_cols):
#Impute using training-set median (numeric) and mode (categorical)
    train = train.copy()
    test = test.copy()

    for col in numeric_cols:
        median_val = train[col].median()
        train[col] = train[col].fillna(median_val)
        test[col] = test[col].fillna(median_val)

    for col in categorical_cols:
        mode_val = train[col].mode()[0]
        train[col] = train[col].fillna(mode_val)
        test[col] = test[col].fillna(mode_val)

    return train, test


global_train, global_test = impute(global_train, global_test, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
male_train, male_test = impute(male_train, male_test, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
female_train, female_test = impute(female_train, female_test, NUMERIC_FEATURES, CATEGORICAL_FEATURES)

def one_hot_encode(train, test, categorical_cols):

    train = pd.get_dummies(train, columns=categorical_cols, drop_first=False)
    test = pd.get_dummies(test, columns=categorical_cols, drop_first=False)

    # Align: add missing columns in test, remove extra columns
    for col in train.columns:
        if col not in test.columns:
            test[col] = 0
    test = test[train.columns]

    return train, test


global_train, global_test = one_hot_encode(global_train, global_test, CATEGORICAL_FEATURES)
male_train, male_test = one_hot_encode(male_train, male_test, CATEGORICAL_FEATURES)
female_train, female_test = one_hot_encode(female_train, female_test, CATEGORICAL_FEATURES)


os.makedirs(OUTPUT_DIR, exist_ok=True)

global_train.to_csv(f"{OUTPUT_DIR}/global_train.csv", index=False)
global_test.to_csv(f"{OUTPUT_DIR}/global_test.csv", index=False)
male_train.to_csv(f"{OUTPUT_DIR}/male_train.csv", index=False)
male_test.to_csv(f"{OUTPUT_DIR}/male_test.csv", index=False)
female_train.to_csv(f"{OUTPUT_DIR}/female_train.csv", index=False)
female_test.to_csv(f"{OUTPUT_DIR}/female_test.csv", index=False)
missing_rates.to_frame("missing_pct").to_csv(f"{OUTPUT_DIR}/missing_rates.csv")

print(f"\nSaved all outputs to {OUTPUT_DIR}/")
