import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

def find_threshold_youden(y_true, y_proba):    #J = sensitivity + specificity - 1
    
    fpr, tpr, thresholds = roc_curve(np.asarray(y_true), np.asarray(y_proba))
    j = tpr - fpr
    best_idx = np.argmax(j)
    return float(thresholds[best_idx])


def compute_overall_metrics(y_true, y_proba, threshold=0.5):
    
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    y_true = np.asarray(y_true)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_proba),
    }


# Subgroup metrics 
def compute_subgroup_metrics(y_true, y_proba, sex_labels, threshold=0.5):

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    sex_labels = np.asarray(sex_labels)
    y_pred = (y_proba >= threshold).astype(int)

    results = {}
    for label, mask in [("male", sex_labels == 0), ("female", sex_labels == 1)]:
        yt, yp, ypr = y_true[mask], y_pred[mask], y_proba[mask]
        rec = recall_score(yt, yp, zero_division=0)
        results[label] = {
            "roc_auc": roc_auc_score(yt, ypr) if len(np.unique(yt)) > 1 else np.nan,
            "recall": rec,
            "fnr": 1.0 - rec,
        }

    return results


def compute_fairness_metrics(y_true, y_proba, sex_labels, threshold=0.5):

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    sex_labels = np.asarray(sex_labels)
    y_pred = (y_proba >= threshold).astype(int)

    sub = compute_subgroup_metrics(y_true, y_proba, sex_labels, threshold)

    fnr_f = sub["female"]["fnr"]
    fnr_m = sub["male"]["fnr"]
    tpr_f = sub["female"]["recall"]
    tpr_m = sub["male"]["recall"]

    # Disparate Impact
    pred_pos_f = y_pred[sex_labels == 1].mean() if (sex_labels == 1).any() else np.nan
    pred_pos_m = y_pred[sex_labels == 0].mean() if (sex_labels == 0).any() else np.nan
    di = pred_pos_f / pred_pos_m if pred_pos_m and pred_pos_m > 0 else np.nan

    return {
        "delta_fnr": fnr_f - fnr_m,
        "eod": tpr_f - tpr_m,
        "di": di,
    }


def evaluate_model(y_true, y_proba, sex_labels, threshold=0.5):

    return {
        "overall": compute_overall_metrics(y_true, y_proba, threshold),
        "subgroup": compute_subgroup_metrics(y_true, y_proba, sex_labels, threshold),
        "fairness": compute_fairness_metrics(y_true, y_proba, sex_labels, threshold),
    }


def collect_nested(records, keys):     # Extract a list of float values from nested dicts by key path
    
    vals = []
    for r in records:
        v = r
        for k in keys:
            v = v[k]
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(v)
    return vals


def bootstrap_metrics(y_true, y_proba, sex_labels, threshold=0.5,            # 95 % CIs for all evaluate_model metrics via bootstrap resampling
                      n_boot=1000, seed=42):

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    sex_labels = np.asarray(sex_labels)
    rng = np.random.RandomState(seed)
    n = len(y_true)

    boot_records = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            result = evaluate_model(y_true[idx], y_proba[idx], sex_labels[idx],
                                    threshold)
            boot_records.append(result)
        except Exception:
            continue

    point = evaluate_model(y_true, y_proba, sex_labels, threshold)

    def _ci(keys):
        vals = collect_nested(boot_records, keys)
        if len(vals) < 2:
            return (point_val(keys), np.nan, np.nan)
        lo = np.percentile(vals, 2.5)
        hi = np.percentile(vals, 97.5)
        return (point_val(keys), lo, hi)

    def point_val(keys):
        v = point
        for k in keys:
            v = v[k]
        return v

    return {
        "overall": {
            m: _ci(["overall", m])
            for m in ["roc_auc", "pr_auc", "accuracy", "precision",
                       "recall", "specificity", "f1", "brier"]
        },
        "subgroup": {
            sex: {
                m: _ci(["subgroup", sex, m])
                for m in ["roc_auc", "recall", "fnr"]
            }
            for sex in ["male", "female"]
        },
        "fairness": {
            m: _ci(["fairness", m])
            for m in ["delta_fnr", "eod", "di"]
        },
    }


def bootstrap_sex_specific_fairness(y_true_m, y_proba_m, threshold_m,
                                    y_true_f, y_proba_f, threshold_f,
                                    n_boot=1000, seed=42):

    y_true_m = np.asarray(y_true_m)
    y_proba_m = np.asarray(y_proba_m)
    y_true_f = np.asarray(y_true_f)
    y_proba_f = np.asarray(y_proba_f)
    rng = np.random.RandomState(seed)
    n_m, n_f = len(y_true_m), len(y_true_f)

    def compute(yt_m, yp_m, yt_f, yp_f):
        pred_m = (yp_m >= threshold_m).astype(int)
        pred_f = (yp_f >= threshold_f).astype(int)
        rec_m = recall_score(yt_m, pred_m, zero_division=0)
        rec_f = recall_score(yt_f, pred_f, zero_division=0)
        fnr_m, fnr_f = 1.0 - rec_m, 1.0 - rec_f
        pp_m = pred_m.mean() if len(pred_m) > 0 else np.nan
        pp_f = pred_f.mean() if len(pred_f) > 0 else np.nan
        di = pp_f / pp_m if pp_m and pp_m > 0 else np.nan
        return {"delta_fnr": fnr_f - fnr_m, "eod": rec_f - rec_m, "di": di}

    point = compute(y_true_m, y_proba_m, y_true_f, y_proba_f)

    boots = []
    for _ in range(n_boot):
        idx_m = rng.choice(n_m, size=n_m, replace=True)
        idx_f = rng.choice(n_f, size=n_f, replace=True)
        try:
            r = compute(y_true_m[idx_m], y_proba_m[idx_m],
                         y_true_f[idx_f], y_proba_f[idx_f])
            boots.append(r)
        except Exception:
            continue

    result = {}
    for key in ["delta_fnr", "eod", "di"]:
        vals = [b[key] for b in boots
                if not (isinstance(b[key], float) and np.isnan(b[key]))]
        if len(vals) < 2:
            result[key] = (point[key], np.nan, np.nan)
        else:
            result[key] = (point[key], np.percentile(vals, 2.5),
                           np.percentile(vals, 97.5))
    return result
