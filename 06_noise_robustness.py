import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_utils import (
    RANDOM_SEED,
    add_measurement_noise,
    build_feature_sets,
    evaluate_classifier,
    load_raw_features,
)
from paths import figure_path, table_path
from plot_utils import setup_chinese_font


NOISE_LEVELS = [0.00, 0.01, 0.03, 0.05, 0.07, 0.10]
METHODS = [
    ("KNN", "raw_iv_curve"),
    ("SVM", "raw_iv_curve"),
    ("RF", "raw_iv_curve"),
    ("MLP", "raw_iv_curve"),
    ("PINN-RF", "pinn_params"),
]


def run_noise_robustness():
    setup_chinese_font()
    x_raw, y = load_raw_features()
    records = []

    for noise in NOISE_LEVELS:
        print(f"\n>>> 噪声水平: {noise * 100:.0f}%")
        if noise == 0:
            x_eval = x_raw
        else:
            x_eval = add_measurement_noise(
                x_raw, noise_level=noise, missing_rate=0.03, random_seed=RANDOM_SEED
            )
        feature_sets = build_feature_sets(x_eval)

        for method, feature_key in METHODS:
            result = evaluate_classifier(method, feature_sets[feature_key], y)
            records.append({
                "noise_level": noise,
                "method": method,
                "accuracy": result["accuracy"],
                "f1_macro": result["f1_macro"],
                "train_time_s": result["train_time"],
            })
            print(f"{method:<8} Acc={result['accuracy'] * 100:.2f}% F1={result['f1_macro'] * 100:.2f}%")

    df = pd.DataFrame(records)
    out_csv = table_path("noise_robustness_results.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 5))
    for method in df["method"].unique():
        sub = df[df["method"] == method]
        plt.plot(sub["noise_level"] * 100, sub["accuracy"] * 100, marker="o", label=method)
    plt.xlabel("噪声水平 (%)")
    plt.ylabel("诊断准确率 (%)")
    plt.title("不同噪声水平下的诊断鲁棒性")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path("fig_noise_robustness_curve.png"), dpi=200, bbox_inches="tight")
    plt.savefig(figure_path("fig_noise_robustness_curve.svg"), format="svg", bbox_inches="tight")
    plt.show()

    print(f"\n✅ 噪声鲁棒性结果已保存: {out_csv}")


if __name__ == "__main__":
    run_noise_robustness()
