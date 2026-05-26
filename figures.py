import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from metrics import bootstrap_metrics, bootstrap_sex_specific_fairness

MODELS = {
    "SCORE2": "predictions/score2_predictions.csv",
    "Logistic Regression": "predictions/lr_predictions.csv",
    "XGBoost": "predictions/xgboost_predictions.csv",
    "TabPFN": "predictions/tabpfn_predictions.csv",
}
RESULTS = {
    "SCORE2": "predictions/score2_results.json",
    "Logistic Regression": "predictions/lr_results.json",
    "XGBoost": "predictions/xgboost_results.json",
    "TabPFN": "predictions/tabpfn_results.json",
}
COLORS = {
    "SCORE2": "#888888",
    "Logistic Regression": "#1f77b4",
    "XGBoost": "#ff7f0e",
    "TabPFN": "#2ca02c",
}
OUT_DIR = "figures"


def load_predictions():
    preds = {}
    for name, path in MODELS.items():
        df = pd.read_csv(path)
        preds[name] = df
    return preds


def load_results():
    results = {}
    for name, path in RESULTS.items():
        with open(path) as f:
            results[name] = json.load(f)
    return results


def figure1(preds, results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for name, df in preds.items():
        y_true = df["y_true"].values
        y_proba = df["y_proba"].values
        auc = results[name]["overall"]["roc_auc"]
        ap = results[name]["overall"]["pr_auc"]
        color = COLORS[name]

        # Panel A: ROC
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        ax1.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {auc:.3f})")

        # Panel B: PR
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ax2.plot(recall, precision, color=color, lw=2, label=f"{name} (AP = {ap:.3f})")

    # ROC formatting
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax1.set_xlabel("False Positive Rate", fontsize=12)
    ax1.set_ylabel("True Positive Rate", fontsize=12)
    ax1.set_title("A. ROC Curves", fontsize=13, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=10)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])

    # PR formatting
    prevalence = preds["TabPFN"]["y_true"].mean()
    ax2.axhline(prevalence, color="k", ls="--", lw=1, alpha=0.5, label=f"Prevalence ({prevalence:.2f})")
    ax2.set_xlabel("Recall", fontsize=12)
    ax2.set_ylabel("Precision", fontsize=12)
    ax2.set_title("B. Precision-Recall Curves", fontsize=13, fontweight="bold")
    ax2.legend(loc="lower left", fontsize=10)
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])

    fig.suptitle("Figure 1: Performance comparison of clinical and machine learning models",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figure1.png", dpi=300, bbox_inches="tight")
    print(f"Saved {OUT_DIR}/figure1.png")
    plt.close(fig)


def figure2(results, preds):
    with open("predictions/tabpfn_sex_specific_comparison.json") as f:
        sex_comp = json.load(f)

    # Bootstrap CIs for Panel A error bars
    boot_ci = {}
    for m in ["SCORE2", "TabPFN"]:
        df = preds[m]
        thresh = results[m].get("threshold", 0.5)
        boot_ci[m] = bootstrap_metrics(
            df["y_true"].values, df["y_proba"].values, df["sex_label"].values,
            threshold=thresh,
        )

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A
    models_fairness = ["SCORE2", "TabPFN"]
    x = np.arange(len(models_fairness))
    width = 0.35

    male_fnr = [results[m]["subgroup"]["male"]["fnr"] for m in models_fairness]
    female_fnr = [results[m]["subgroup"]["female"]["fnr"] for m in models_fairness]

    # CI error bars: distance from point estimate to lo/hi
    male_fnr_err = [
        [male_fnr[i] - boot_ci[m]["subgroup"]["male"]["fnr"][1],
         boot_ci[m]["subgroup"]["male"]["fnr"][2] - male_fnr[i]]
        for i, m in enumerate(models_fairness)
    ]
    female_fnr_err = [
        [female_fnr[i] - boot_ci[m]["subgroup"]["female"]["fnr"][1],
         boot_ci[m]["subgroup"]["female"]["fnr"][2] - female_fnr[i]]
        for i, m in enumerate(models_fairness)
    ]
    male_fnr_yerr = np.array(male_fnr_err).T.clip(min=0)
    female_fnr_yerr = np.array(female_fnr_err).T.clip(min=0)

    bars1 = ax1.bar(x - width / 2, male_fnr, width, label="Male", color="#5b9bd5",
                    yerr=male_fnr_yerr, capsize=4, error_kw={"lw": 1.2})
    bars2 = ax1.bar(x + width / 2, female_fnr, width, label="Female", color="#ed7d31",
                    yerr=female_fnr_yerr, capsize=4, error_kw={"lw": 1.2})

    ax1.set_ylabel("False Negative Rate", fontsize=11)
    ax1.set_title("A. FNR by Sex", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_fairness, fontsize=10)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 1.15)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.3f}",
                     ha="center", va="bottom", fontsize=9)

    # Panel B: ΔFNR
    delta_fnr = [results[m]["fairness"]["delta_fnr"] for m in models_fairness]

    bars = ax2.bar(models_fairness, delta_fnr, color=["#888888", "#2ca02c"], width=0.5)
    ax2.set_ylabel("ΔFNR (Female − Male)", fontsize=11)
    ax2.set_title("B. FNR Disparity", fontsize=13, fontweight="bold")
    ax2.axhline(0, color="k", lw=0.8)

    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.3f}",
                 ha="center", va="bottom", fontsize=9)

    # Panel C
    labels = ["Global\n(Male subset)", "Male-\nSpecific", "Global\n(Female subset)", "Female-\nSpecific"]
    fnr_vals = [
        results["TabPFN"]["subgroup"]["male"]["fnr"],
        1.0 - sex_comp["male"]["sex_specific"]["recall"],
        results["TabPFN"]["subgroup"]["female"]["fnr"],
        1.0 - sex_comp["female"]["sex_specific"]["recall"],
    ]
    bar_colors = ["#2ca02c", "#90d090", "#2ca02c", "#90d090"]

    bars = ax3.bar(labels, fnr_vals, color=bar_colors, width=0.6, edgecolor="black", linewidth=0.5)
    ax3.set_ylabel("False Negative Rate", fontsize=11)
    ax3.set_title("C. Sex-Specific Training Effect", fontsize=13, fontweight="bold")
    ax3.set_ylim(0, max(fnr_vals) * 1.4 if max(fnr_vals) > 0 else 1.15)

    for bar in bars:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.3f}",
                 ha="center", va="bottom", fontsize=9)

    fig.suptitle("Figure 2: Subgroup fairness and impact of sex-specific modeling",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figure2.png", dpi=300, bbox_inches="tight")
    print(f"Saved {OUT_DIR}/figure2.png")
    plt.close(fig)


def table1(results, preds):
    with open("predictions/tabpfn_sex_specific_comparison.json") as f:
        sex_comp = json.load(f)

    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        return f"{v:.4f}"

    def fmt_ci(ci_tuple):
        """Format a (point, lo, hi) tuple as 'point [lo, hi]'."""
        pt, lo, hi = ci_tuple
        if pt is None or (isinstance(pt, float) and np.isnan(pt)):
            return "N/A"
        if np.isnan(lo) or np.isnan(hi):
            return fmt(pt)
        return f"{pt:.4f} [{lo:.4f}, {hi:.4f}]"

    # Bootstrap CIs for each model (using model-specific thresholds)
    boot = {}
    for name in ["SCORE2", "Logistic Regression", "XGBoost", "TabPFN"]:
        df = preds[name]
        thresh = results[name].get("threshold", 0.5)
        boot[name] = bootstrap_metrics(
            df["y_true"].values, df["y_proba"].values, df["sex_label"].values,
            threshold=thresh,
        )

    rows = []
    for name in ["SCORE2", "Logistic Regression", "XGBoost", "TabPFN"]:
        r = results[name]
        b = boot[name]
        rows.append({
            "Model": name,
            "Threshold": fmt(r.get("threshold", 0.5)),
            "AUC": fmt(r["overall"]["roc_auc"]),
            "Recall": fmt(r["overall"]["recall"]),
            "FNR": fmt(1.0 - r["overall"]["recall"]),
            "Brier": fmt(r["overall"]["brier"]),
            "ΔFNR": fmt_ci(b["fairness"]["delta_fnr"]),
            "EOD": fmt_ci(b["fairness"]["eod"]),
            "DI": fmt_ci(b["fairness"]["di"]),
        })

    # Best ML (Global) = TabPFN (duplicate row with label)
    r = results["TabPFN"]
    b = boot["TabPFN"]
    rows.append({
        "Model": "Best ML (Global)",
        "Threshold": fmt(r.get("threshold", 0.5)),
        "AUC": fmt(r["overall"]["roc_auc"]),
        "Recall": fmt(r["overall"]["recall"]),
        "FNR": fmt(1.0 - r["overall"]["recall"]),
        "Brier": fmt(r["overall"]["brier"]),
        "ΔFNR": fmt_ci(b["fairness"]["delta_fnr"]),
        "EOD": fmt_ci(b["fairness"]["eod"]),
        "DI": fmt_ci(b["fairness"]["di"]),
    })

    # Best ML (Sex-Specific): combined metrics from sex-specific models
    m_metrics = sex_comp["male"]["sex_specific"]
    f_metrics = sex_comp["female"]["sex_specific"]
    n_m, n_f = 130, 34
    n_total = n_m + n_f

    combined_recall = (m_metrics["recall"] * n_m + f_metrics["recall"] * n_f) / n_total
    combined_fnr = 1.0 - combined_recall
    delta_fnr_ss = (1.0 - f_metrics["recall"]) - (1.0 - m_metrics["recall"])
    eod_ss = f_metrics["recall"] - m_metrics["recall"]
    pred_m = pd.read_csv("predictions/tabpfn_male_predictions.csv")
    pred_f = pd.read_csv("predictions/tabpfn_female_predictions.csv")
    m_thresh = sex_comp["male"].get("sex_specific_threshold", 0.5)
    f_thresh = sex_comp["female"].get("sex_specific_threshold", 0.5)
    pred_pos_m = (pred_m["y_proba_sex_specific"] >= m_thresh).mean()
    pred_pos_f = (pred_f["y_proba_sex_specific"] >= f_thresh).mean()
    di_ss = pred_pos_f / pred_pos_m if pred_pos_m > 0 else np.nan

    combined_brier = (m_metrics["brier"] * n_m + f_metrics["brier"] * n_f) / n_total

    # Bootstrap CIs for sex-specific fairness metrics
    boot_ss = bootstrap_sex_specific_fairness(
        pred_m["y_true"].values, pred_m["y_proba_sex_specific"].values, m_thresh,
        pred_f["y_true"].values, pred_f["y_proba_sex_specific"].values, f_thresh,
    )

    rows.append({
        "Model": "Best ML (Sex-Specific)",
        "Threshold": f"M:{m_thresh:.2f}/F:{f_thresh:.2f}",
        "AUC": fmt((m_metrics["roc_auc"] * n_m + f_metrics["roc_auc"] * n_f) / n_total),
        "Recall": fmt(combined_recall),
        "FNR": fmt(combined_fnr),
        "Brier": fmt(combined_brier),
        "ΔFNR": fmt_ci(boot_ss["delta_fnr"]),
        "EOD": fmt_ci(boot_ss["eod"]),
        "DI": fmt_ci(boot_ss["di"]),
    })

    df = pd.DataFrame(rows)

    # Save CSV
    df.to_csv(f"{OUT_DIR}/table1.csv", index=False)
    print(f"\nSaved {OUT_DIR}/table1.csv")


if __name__ == "__main__":
    preds = load_predictions()
    results = load_results()

    figure1(preds, results)
    figure2(results, preds)
    table1(results, preds)


