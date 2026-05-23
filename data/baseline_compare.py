import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiment_utils import build_feature_sets, evaluate_classifier, load_raw_features
from paths import figure_path
from plot_utils import setup_chinese_font


print("正在加载数据集...")
setup_chinese_font()
x_raw, y = load_raw_features()
feature_sets = build_feature_sets(x_raw)
print(f"✅ 数据加载成功！共计 {len(y)} 个样本。")

method_configs = [
    ("KNN", "raw_iv_curve", "KNN"),
    ("SVM", "raw_iv_curve", "SVM"),
    ("MLP", "raw_iv_curve", "MLP"),
    ("PINN-RF", "pinn_params", "PINN-RF"),
]

results = {
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1-score": [],
}
model_names = []

print("\n开始评估对比模型...")
for display_name, feature_key, method in method_configs:
    result = evaluate_classifier(method, feature_sets[feature_key], y)
    cm = result["confusion_matrix"]
    per_class_precision = np.divide(
        cm.diagonal(), cm.sum(axis=0), out=np.zeros(cm.shape[0], dtype=float), where=cm.sum(axis=0) != 0
    )
    per_class_recall = np.divide(
        cm.diagonal(), cm.sum(axis=1), out=np.zeros(cm.shape[0], dtype=float), where=cm.sum(axis=1) != 0
    )
    precision = float(np.mean(per_class_precision))
    recall = float(np.mean(per_class_recall))

    model_names.append(display_name)
    results["Accuracy"].append(result["accuracy"])
    results["Precision"].append(precision)
    results["Recall"].append(recall)
    results["F1-score"].append(result["f1_macro"])
    print(
        f"[{display_name}] Acc: {result['accuracy']:.4f}, "
        f"Pre: {precision:.4f}, Rec: {recall:.4f}, F1: {result['f1_macro']:.4f}"
    )

print("\n正在生成对比柱状图...")
fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
x = np.arange(len(model_names))
width = 0.2
colors = ["#4A90E2", "#50E3C2", "#B8E986", "#F5A623"]

bars = []
for idx, metric in enumerate(results.keys()):
    offset = (idx - 1.5) * width
    bars.append(
        ax.bar(
            x + offset,
            results[metric],
            width,
            label=metric,
            color=colors[idx],
            edgecolor="black",
            linewidth=0.5,
        )
    )

ax.set_ylabel("评估指标得分", fontsize=12)
ax.set_title("不同诊断模型分类性能综合对比", fontsize=15, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=12, fontweight="bold")
ax.set_ylim(0.5, 1.08)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.7)

for rects in bars:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height * 100:.2f}%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

plt.tight_layout()
save_img_path = figure_path("baseline_comparison_chart_percent.png")
plt.savefig(save_img_path, bbox_inches="tight", dpi=300)
print(f"✅ 对比图表已保存为: {save_img_path}")
plt.show()
