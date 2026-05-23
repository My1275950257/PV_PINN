import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import pandas as pd

from experiment_utils import build_feature_sets, evaluate_classifier, load_raw_features
from paths import figure_path, table_path
from plot_utils import setup_chinese_font


ABLATIONS = [
    ("Raw I-V + RF", "raw_iv_curve", "RF"),
    ("I-V Features + RF", "iv_features", "RF"),
    ("PINN Params + RF", "pinn_params", "RF"),
    ("PINN Params + I-V Features + RF", "pinn_iv_fused", "PINN-RF"),
    ("PINN Params + I-V Features + SVM", "pinn_iv_fused", "SVM"),
]


def run_ablation_study():
    setup_chinese_font()
    x_raw, y = load_raw_features()
    feature_sets = build_feature_sets(x_raw)
    records = []

    print(">>> 消融实验...")
    for label, feature_key, method in ABLATIONS:
        result = evaluate_classifier(method, feature_sets[feature_key], y)
        records.append({
            "setting": label,
            "feature_set": feature_key,
            "classifier": method,
            "accuracy": result["accuracy"],
            "f1_macro": result["f1_macro"],
            "train_time_s": result["train_time"],
        })
        print(f"{label:<38} Acc={result['accuracy'] * 100:.2f}% F1={result['f1_macro'] * 100:.2f}%")

    df = pd.DataFrame(records)
    out_csv = table_path("ablation_study_results.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 5))
    bars = plt.bar(df["setting"], df["accuracy"] * 100, color="#4C78A8")
    for bar, value in zip(bars, df["accuracy"] * 100):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.3, f"{value:.2f}%",
                 ha="center", va="bottom", fontsize=9)
    plt.ylabel("诊断准确率 (%)")
    plt.title("PINN-RF 各模块消融实验")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(max(0, min(df["accuracy"] * 100) - 5), 101)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(figure_path("fig_ablation_study.png"), dpi=200, bbox_inches="tight")
    plt.savefig(figure_path("fig_ablation_study.svg"), format="svg", bbox_inches="tight")
    plt.show()

    print(f"\n✅ 消融实验结果已保存: {out_csv}")


if __name__ == "__main__":
    run_ablation_study()
