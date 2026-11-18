"""
Visualization and Diagnostic Plotting for PathSnap Model Comparison
Generates compulsory comparative charts:
1. Cross-validation Macro F1 comparison across all models
2. Student-testing Top-1 Accuracy and Macro F1 comparison
3. Log Loss comparison across models
4. Inference Latency comparison
5. Confusion Matrices for models
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_model_comparison_plots(results_csv: str = "results/model_comparison.csv", output_dir: str = "results/plots"):
    if not os.path.exists(results_csv):
        print(f"Results CSV not found: {results_csv}")
        return

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(results_csv)

    sns.set_theme(style="whitegrid", font_scale=1.1)
    palette = sns.color_palette("muted")

    # 1. Cross-Validation Macro F1-score comparison
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=df, x="model", y="cv_macro_f1", palette="Blues_d")
    plt.title("5-Fold GroupKFold Cross-Validation Macro F1-Score Across Models", fontweight="bold")
    plt.xlabel("Model")
    plt.ylabel("CV Macro F1-Score")
    plt.ylim(0.5, 1.0)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 7), textcoords='offset points', fontweight='bold', fontsize=10)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cv_macro_f1_comparison.png"), dpi=200)
    plt.close()

    # 2. Student-Testing Top-1 Accuracy vs. Macro F1 comparison
    plt.figure(figsize=(11, 5))
    df_melt = df.melt(id_vars=["model"], value_vars=["top1_accuracy", "student_test_macro_f1"],
                      var_name="Metric", value_name="Score")
    df_melt["Metric"] = df_melt["Metric"].replace({
        "top1_accuracy": "Observation Top-1 Accuracy",
        "student_test_macro_f1": "Candidate Macro F1"
    })
    ax = sns.barplot(data=df_melt, x="model", y="Score", hue="Metric", palette="Set2")
    plt.title("Student-Testing Performance: Top-1 Accuracy & Macro F1-Score", fontweight="bold")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0.4, 1.0)
    for p in ax.patches:
        val = p.get_height()
        if not np.isnan(val) and val > 0:
            ax.annotate(f"{val:.3f}", (p.get_x() + p.get_width() / 2., val),
                        ha='center', va='center', xytext=(0, 6), textcoords='offset points', fontsize=9)
    plt.xticks(rotation=20, ha="right")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "student_top1_vs_f1_comparison.png"), dpi=200)
    plt.close()

    # 3. Log Loss Comparison (lower is better)
    plt.figure(figsize=(9, 5))
    df_loss = df.dropna(subset=["log_loss"])
    ax = sns.barplot(data=df_loss, x="model", y="log_loss", palette="Reds_d")
    plt.title("Log Loss on Student-Testing Dataset (Lower is Better)", fontweight="bold")
    plt.xlabel("Model")
    plt.ylabel("Binary Cross-Entropy Log Loss")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 6), textcoords='offset points', fontweight='bold', fontsize=10)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "log_loss_comparison.png"), dpi=200)
    plt.close()

    # 4. Inference Latency (ms per 1,000 observations)
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=df, x="model", y="inference_time_ms_per_1k", palette="Greens_d")
    plt.title("Inference Latency (Milliseconds per 1,000 Observations)", fontweight="bold")
    plt.xlabel("Model")
    plt.ylabel("Latency (ms / 1k observations)")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f} ms", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 6), textcoords='offset points', fontsize=10)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "inference_latency_comparison.png"), dpi=200)
    plt.close()

    print(f"Model comparison plots successfully generated in: {output_dir}")


if __name__ == "__main__":
    generate_model_comparison_plots()
