import matplotlib.pyplot as plt
import numpy as np
from paths import figure_path
from plot_utils import setup_chinese_font

# 设置中文字体
setup_chinese_font()

# --- 数据准备 (已删除“（本文）”) ---
methods = ['KNN', 'SVM', '纯RF', 'MLP', 'PINN-RF']
overall_acc = [78.21, 94.38, 89.83, 92.83, 95.71]

categories = ['正常', '老化', 'PID衰减', '二极管短路', '局部阴影', '热斑']
data_7b = {
    'KNN': [78.0, 88.0, 73.0, 83.0, 63.0, 84.0],
    'SVM': [95.0, 96.0, 94.0, 98.0, 88.0, 95.0],
    '纯RF': [91.0, 94.0, 90.0, 94.0, 78.0, 92.0],
    'MLP': [93.0, 95.0, 92.0, 97.0, 85.0, 95.0],
    'PINN-RF': [98.0, 98.0, 94.0, 99.0, 89.0, 96.0]
}

# 配色方案
colors = ['#8EBCDA', '#66B08A', '#E4B632', '#C86B6B', '#E54D42']

# 创建画布，增加一点高度
fig = plt.figure(figsize=(12, 11))

# --- 绘制图 7a 综合诊断准确率对比 ---
ax1 = fig.add_subplot(2, 1, 1)
bars1 = ax1.bar(methods, overall_acc, color=colors, width=0.5)

# 调整细节：删除标题，调大字体，增加上限防止溢出
ax1.set_ylim(75, 110)  # 上限调至110%
ax1.set_yticks(np.arange(75, 105, 5))
ax1.set_yticklabels([f'{x}%' for x in np.arange(75, 105, 5)], fontsize=12)
ax1.set_xticks(np.arange(len(methods)))
ax1.set_xticklabels(methods, fontsize=13)
ax1.grid(axis='y', linestyle='-', alpha=0.3)

# 添加数值标注 - 调大字号
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height + 1.0, f'{height}%',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

# --- 绘制图 7b 各类别识别准确率对比 ---
ax2 = fig.add_subplot(2, 1, 2)
x = np.arange(len(categories))
width = 0.15

for i, method in enumerate(methods):
    offset = (i - 2) * width
    bars2 = ax2.bar(x + offset, data_7b[method], width, label=method, color=colors[i])

# 调整细节：删除标题，调大字体，增加上限给图例留空间
ax2.set_xticks(x)
ax2.set_xticklabels(categories, fontsize=13)
ax2.set_ylim(60, 120)  # 上限调至120%，防止顶部的图例遮挡柱子
ax2.set_yticks(np.arange(60, 110, 10))
ax2.set_yticklabels([f'{x}%' for x in np.arange(60, 110, 10)], fontsize=12)
ax2.grid(axis='y', linestyle='-', alpha=0.3)

# 图例设置 - 调大字号
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=5, frameon=False, fontsize=12)

plt.tight_layout(pad=3.0) # 增加子图间距
plt.savefig(figure_path('manual_comparison_plot.png'), dpi=200, bbox_inches='tight')
plt.show()
