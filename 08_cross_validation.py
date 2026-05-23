import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from experiment_utils import build_feature_sets, load_raw_features, make_classifier
from paths import table_path


SEEDS = [0, 7, 21, 42, 100]
METHODS = [
    ("KNN", "raw_iv_curve"),
    ("SVM", "raw_iv_curve"),
    ("RF", "raw_iv_curve"),
    ("MLP", "raw_iv_curve"),
    ("PINN-RF", "pinn_params"),
]


def run_cross_validation():
    x_raw, y = load_raw_features()
    feature_sets = build_feature_sets(x_raw)
    records = []

    print(">>> 多随机种子 5 折交叉验证...")
    for seed in SEEDS:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for method, feature_key in METHODS:
            clf = make_classifier(method, random_seed=seed)
            scores = cross_validate(
                clf,
                feature_sets[feature_key],
                y,
                cv=cv,
                scoring=("accuracy", "f1_macro"),
                n_jobs=-1,
                return_train_score=False,
            )
            for fold_idx, (acc, f1) in enumerate(zip(scores["test_accuracy"], scores["test_f1_macro"]), 1):
                records.append({
                    "seed": seed,
                    "fold": fold_idx,
                    "method": method,
                    "feature_set": feature_key,
                    "accuracy": acc,
                    "f1_macro": f1,
                })
            print(f"seed={seed:<3} {method:<8} Acc={scores['test_accuracy'].mean() * 100:.2f}%")

    df = pd.DataFrame(records)
    detail_csv = table_path("cross_validation_detail.csv")
    summary_csv = table_path("cross_validation_summary.csv")
    df.to_csv(detail_csv, index=False, encoding="utf-8-sig")

    summary = df.groupby("method").agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        f1_macro_mean=("f1_macro", "mean"),
        f1_macro_std=("f1_macro", "std"),
    ).reset_index()
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\n===== mean ± std =====")
    for _, row in summary.iterrows():
        print(
            f"{row['method']:<8} "
            f"Acc={row['accuracy_mean'] * 100:.2f}±{row['accuracy_std'] * 100:.2f}%  "
            f"F1={row['f1_macro_mean'] * 100:.2f}±{row['f1_macro_std'] * 100:.2f}%"
        )
    print(f"\n✅ 交叉验证明细已保存: {detail_csv}")
    print(f"✅ 交叉验证汇总已保存: {summary_csv}")


if __name__ == "__main__":
    run_cross_validation()
