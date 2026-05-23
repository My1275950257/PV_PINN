import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiment_utils import PINN_FEATURE_NAMES, build_feature_sets, load_raw_features
from paths import figure_path, table_path
from plot_utils import setup_chinese_font


def run_feature_importance():
    setup_chinese_font()
    x_raw, y = load_raw_features()
    x_pinn = build_feature_sets(x_raw)["pinn_params"]

    x_train, x_test, y_train, y_test = train_test_split(
        x_pinn, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    clf.fit(x_train, y_train)
    acc = clf.score(x_test, y_test)
    rf = clf.named_steps["rf"]

    df = pd.DataFrame({
        "feature": PINN_FEATURE_NAMES,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    out_csv = table_path("feature_importance.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    top = df.head(15).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["importance"], color="#E45756")
    plt.xlabel("随机森林特征重要性")
    plt.title(f"PINN-RF 物理参数重要性 (Acc={acc * 100:.2f}%)")
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(figure_path("fig_pinn_param_importance.png"), dpi=200, bbox_inches="tight")
    plt.savefig(figure_path("fig_pinn_param_importance.svg"), format="svg", bbox_inches="tight")
    plt.show()

    print(f"✅ 特征重要性已保存: {out_csv}")
    print(df.head(15).to_string(index=False))


if __name__ == "__main__":
    run_feature_importance()
