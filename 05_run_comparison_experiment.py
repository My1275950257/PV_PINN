"""
comparison_experiment_v3.py  —— 4.3节完整对比实验
============================================================
对比维度（体现PINN-RF综合优势）：
  1. 理想 vs 含噪声条件综合准确率对比
  2. 各方法噪声鲁棒性下降幅度
  3. 小样本条件下各方法准确率（每类20/50/100/200样本）
  4. 各方法在噪声条件下的各类别识别率热力图
  5. 综合能力雷达图（准确率/鲁棒性/可解释性/小样本/训练开销）
  6. 汇总对比表（含F1、训练时间）

运行前提：
   python 01_generate_data.py   → 生成 ./data/train_dataset.npz
   python 02_train_pinn.py      → 生成 pinn_model_v3.pth
"""

import os, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import seaborn as sns

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score,
                              f1_score, classification_report)

from pinn_model import PV_PINN
from iv_features import extract_iv_curve_features
from paths import figure_path

# ─────────── 全局字体 ───────────
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ─────────── 常量 ───────────
DATA_PATH   = './data/train_dataset.npz'
MODEL_PATH  = 'pinn_model_v3.pth'
CLASS_NAMES = ['正常', '老化', 'PID衰减', '二极管短路', '局部阴影', '热斑']
RANDOM_SEED = 42

# 各方法颜色
COLORS = {
    'KNN':          '#5B8DBE',
    'SVM':          '#4CAF8A',
    '纯RF':          '#E0A830',
    'MLP':          '#9B6BB5',
    'PINN-RF(本文)': '#E24B4A',
}
METHOD_LIST = ['KNN', 'SVM', '纯RF', 'MLP', 'PINN-RF(本文)']


# ═══════════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════════

def add_measurement_noise(X_raw, noise_level=0.05, missing_rate=0.03):
    """模拟真实I-V采集噪声（高斯 + 随机毛刺）"""
    X_noisy = X_raw.copy()
    N, D = X_noisy.shape
    noise_std = noise_level * np.max(np.abs(X_raw), axis=1, keepdims=True)
    X_noisy += np.random.normal(0, 1, X_noisy.shape) * noise_std
    mask = np.random.rand(N, D) < missing_rate
    X_noisy[mask] *= np.random.uniform(0.7, 1.3, mask.sum())
    return np.clip(X_noisy, -0.1, 1.1)


def load_raw_features():
    data  = np.load(DATA_PATH)
    X_V   = data['V'] / 60.0
    X_I   = data['I'] / 16.0
    X_raw = np.concatenate([X_V, X_I], axis=1)
    Y     = data['label']
    return X_raw, Y


def extract_pinn_features(X_raw):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PV_PINN(input_dim=200).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    X_tensor = torch.tensor(X_raw, dtype=torch.float32)
    loader   = DataLoader(TensorDataset(X_tensor), batch_size=512, shuffle=False)
    feats = []
    with torch.no_grad():
        for (batch,) in loader:
            feats.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(feats, axis=0)


def build_fused_features(X_raw, X_phys):
    """Fuse PINN physical parameters with compact I-V electrical descriptors."""
    X_curve = extract_iv_curve_features(X_raw)
    return np.concatenate([X_phys, X_curve], axis=1)


def build_classifiers(random_seed=RANDOM_SEED):
    """返回分类器列表 (name, clf, use_phys_feat)"""
    return [
        ('KNN',   KNeighborsClassifier(n_neighbors=7, metric='euclidean', n_jobs=-1), False),
        ('SVM',   SVC(kernel='rbf', C=10, gamma='scale', random_state=random_seed),   False),
        ('纯RF',   RandomForestClassifier(n_estimators=100, max_depth=10,
                                          random_state=random_seed, n_jobs=-1),        False),
        ('MLP',   MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu',
                                max_iter=500, random_state=random_seed,
                                early_stopping=True, validation_fraction=0.1),         False),
        ('PINN-RF(本文)', RandomForestClassifier(n_estimators=200, max_depth=8,
                                                  random_state=random_seed, n_jobs=-1), True),
    ]


# ═══════════════════════════════════════════════════════════════════
#  实验1：理想 vs 含噪声 综合准确率
# ═══════════════════════════════════════════════════════════════════

def run_accuracy_experiment(X_raw, X_raw_noisy, X_phys, X_phys_noisy, Y):
    """返回 res_clean, res_noisy, timing_clean"""
    results_clean, results_noisy, timings = {}, {}, {}

    sc = StandardScaler()
    sc_n = StandardScaler()

    X_tr_r,  X_te_r,  y_tr, y_te = train_test_split(X_raw, Y, test_size=0.2,
                                                      random_state=RANDOM_SEED, stratify=Y)
    X_tr_rn, X_te_rn, _,    _    = train_test_split(X_raw_noisy, Y, test_size=0.2,
                                                      random_state=RANDOM_SEED, stratify=Y)
    X_tr_p,  X_te_p,  _,    _    = train_test_split(X_phys, Y, test_size=0.2,
                                                      random_state=RANDOM_SEED, stratify=Y)
    X_tr_pn, X_te_pn, _,    _    = train_test_split(X_phys_noisy, Y, test_size=0.2,
                                                      random_state=RANDOM_SEED, stratify=Y)

    X_tr_rs  = sc.fit_transform(X_tr_r);   X_te_rs  = sc.transform(X_te_r)
    X_tr_rsn = sc_n.fit_transform(X_tr_rn); X_te_rsn = sc_n.transform(X_te_rn)
    sp = StandardScaler(); X_tr_ps = sp.fit_transform(X_tr_p);   X_te_ps = sp.transform(X_te_p)
    spn = StandardScaler(); X_tr_psn = spn.fit_transform(X_tr_pn); X_te_psn = spn.transform(X_te_pn)

    clf_configs = [
        ('KNN',          KNeighborsClassifier(n_neighbors=7, n_jobs=-1), X_tr_rs,  X_te_rs,  X_tr_rsn, X_te_rsn),
        ('SVM',          SVC(kernel='rbf', C=10, gamma='scale', random_state=RANDOM_SEED),
                         X_tr_rs,  X_te_rs,  X_tr_rsn, X_te_rsn),
        ('纯RF',          RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1),
                         X_tr_r,   X_te_r,   X_tr_rn,  X_te_rn),
        ('MLP',          MLPClassifier(hidden_layer_sizes=(256,128,64), max_iter=500,
                                       random_state=RANDOM_SEED, early_stopping=True),
                         X_tr_rs,  X_te_rs,  X_tr_rsn, X_te_rsn),
        ('PINN-RF(本文)', RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_SEED, n_jobs=-1),
                         X_tr_ps,  X_te_ps,  X_tr_psn, X_te_psn),
    ]

    for item in clf_configs:
        name = item[0]; clf = item[1]
        Xtr_c, Xte_c, Xtr_n, Xte_n = item[2], item[3], item[4], item[5]

        t0 = time.time(); clf.fit(Xtr_c, y_tr); t1 = time.time()
        timings[name] = round(t1 - t0, 3)

        yp_c = clf.predict(Xte_c)
        acc_c = accuracy_score(y_te, yp_c)
        f1_c  = f1_score(y_te, yp_c, average='macro')
        cm_c  = confusion_matrix(y_te, yp_c)
        pca_c = cm_c.diagonal() / cm_c.sum(axis=1)
        results_clean[name] = (acc_c, f1_c, pca_c, cm_c)

        clf2 = type(clf)(**clf.get_params())
        clf2.fit(Xtr_n, y_tr)
        yp_n = clf2.predict(Xte_n)
        acc_n = accuracy_score(y_te, yp_n)
        f1_n  = f1_score(y_te, yp_n, average='macro')
        cm_n  = confusion_matrix(y_te, yp_n)
        pca_n = cm_n.diagonal() / cm_n.sum(axis=1)
        results_noisy[name] = (acc_n, f1_n, pca_n, cm_n)

        print(f"[{name}] 理想:{acc_c*100:.2f}%  噪声:{acc_n*100:.2f}%  训练:{timings[name]}s")

    return results_clean, results_noisy, timings


# ═══════════════════════════════════════════════════════════════════
#  实验2：小样本鲁棒性
# ═══════════════════════════════════════════════════════════════════

def run_few_shot_experiment(X_raw, X_phys, Y, sample_sizes=(20, 50, 100, 150, 200)):
    """每类取 n 个样本，测试各方法准确率随样本量的变化"""
    print("\n>>> 小样本实验...")
    few_shot_results = {m: [] for m in METHOD_LIST}
    n_classes = len(np.unique(Y))

    for n_per_class in sample_sizes:
        idx_sel = []
        for c in range(n_classes):
            idx_c = np.where(Y == c)[0]
            chosen = np.random.choice(idx_c, min(n_per_class, len(idx_c)), replace=False)
            idx_sel.extend(chosen)
        idx_sel = np.array(idx_sel)

        X_sub_r = X_raw[idx_sel]
        X_sub_p = X_phys[idx_sel]
        Y_sub   = Y[idx_sel]

        if len(np.unique(Y_sub)) < 2:
            for m in METHOD_LIST:
                few_shot_results[m].append(np.nan)
            continue

        X_tr_r, X_te_r, y_tr, y_te = train_test_split(X_sub_r, Y_sub, test_size=0.3,
                                                        random_state=RANDOM_SEED, stratify=Y_sub)
        X_tr_p, X_te_p, _,    _    = train_test_split(X_sub_p, Y_sub, test_size=0.3,
                                                        random_state=RANDOM_SEED, stratify=Y_sub)
        sc = StandardScaler()
        X_tr_rs = sc.fit_transform(X_tr_r); X_te_rs = sc.transform(X_te_r)
        sp = StandardScaler()
        X_tr_ps = sp.fit_transform(X_tr_p); X_te_ps = sp.transform(X_te_p)

        for name, clf, use_phys in build_classifiers():
            Xtr = X_tr_ps if use_phys else X_tr_rs
            Xte = X_te_ps if use_phys else X_te_rs
            clf.fit(Xtr, y_tr)
            acc = accuracy_score(y_te, clf.predict(Xte))
            few_shot_results[name].append(acc * 100)

        print(f"  每类{n_per_class}样本: " +
              " | ".join(f"{m}={few_shot_results[m][-1]:.1f}%" for m in METHOD_LIST))

    return few_shot_results, list(sample_sizes)


# ═══════════════════════════════════════════════════════════════════
#  绘图函数
# ═══════════════════════════════════════════════════════════════════

def plot_dual_accuracy(res_clean, res_noisy):
    """图1：理想 vs 噪声 综合准确率对比柱状图"""
    methods   = METHOD_LIST
    acc_clean = [res_clean[m][0] * 100 for m in methods]
    acc_noisy = [res_noisy[m][0] * 100 for m in methods]
    x = np.arange(len(methods)); w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors_list = [COLORS[m] for m in methods]

    bars_c = ax.bar(x - w/2, acc_clean, width=w,
                    color=[c + 'BB' for c in colors_list],
                    label='无测量噪声（理想条件）', zorder=3, edgecolor='white', linewidth=1.2)
    bars_n = ax.bar(x + w/2, acc_noisy, width=w,
                    color=colors_list,
                    label='含测量噪声（现实条件）', zorder=3, edgecolor='white', linewidth=1.2,
                    hatch='////')

    for bar in list(bars_c) + list(bars_n):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8.5)

    bars_c[-1].set_edgecolor('#900000'); bars_c[-1].set_linewidth(2.5)
    bars_n[-1].set_edgecolor('#900000'); bars_n[-1].set_linewidth(2.5)

    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylim([55, 108])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel('综合诊断准确率', fontsize=13)
    ax.set_title('不同测量条件下各方法综合诊断准确率对比', fontsize=14, pad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.45, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=11, loc='lower right')
    plt.tight_layout()
    plt.savefig(figure_path('fig_dual_accuracy.svg'), format='svg', bbox_inches='tight')
    plt.savefig(figure_path('fig_dual_accuracy.png'), dpi=200, bbox_inches='tight')
    print("✅ 已保存: fig_dual_accuracy.svg/png")
    plt.show()


def plot_robustness_drop(res_clean, res_noisy):
    """图2：噪声引入后准确率下降幅度"""
    methods = METHOD_LIST
    drops   = [(res_clean[m][0] - res_noisy[m][0]) * 100 for m in methods]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bar_colors = [COLORS[m] + 'CC' for m in methods]
    bar_colors[-1] = COLORS['PINN-RF(本文)']

    bars = ax.bar(methods, drops, color=bar_colors, zorder=3, width=0.55,
                  edgecolor='white', linewidth=1.2)
    bars[-1].set_edgecolor('#900000'); bars[-1].set_linewidth(2.5)

    for bar, d in zip(bars, drops):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f'↓{d:.2f}pp', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylim([0, max(drops) * 1.4])
    ax.set_ylabel('准确率下降幅度（百分点）', fontsize=13)
    ax.set_xlabel('对比方法', fontsize=13)
    ax.set_title('引入传感器噪声后各方法准确率下降幅度\n（下降越小，鲁棒性越强）', fontsize=13, pad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.45, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(figure_path('fig_robustness_drop.svg'), format='svg', bbox_inches='tight')
    plt.savefig(figure_path('fig_robustness_drop.png'), dpi=200, bbox_inches='tight')
    print("✅ 已保存: fig_robustness_drop.svg/png")
    plt.show()


def plot_few_shot(few_shot_results, sample_sizes):
    """图3：小样本准确率折线图"""
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {'KNN': 'o', 'SVM': 's', '纯RF': '^', 'MLP': 'D', 'PINN-RF(本文)': '*'}

    for name in METHOD_LIST:
        vals = few_shot_results[name]
        lw = 2.5 if name == 'PINN-RF(本文)' else 1.6
        ms = 10  if name == 'PINN-RF(本文)' else 7
        ax.plot(sample_sizes, vals, marker=markers[name], color=COLORS[name],
                linewidth=lw, markersize=ms, label=name, zorder=3)

    ax.set_xlabel('每类训练样本数（个）', fontsize=13)
    ax.set_ylabel('诊断准确率（%）', fontsize=13)
    ax.set_title('各方法在不同训练样本量下的诊断准确率', fontsize=14, pad=10)
    ax.set_xticks(sample_sizes)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(linestyle='--', alpha=0.45, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(figure_path('fig_few_shot.svg'), format='svg', bbox_inches='tight')
    plt.savefig(figure_path('fig_few_shot.png'), dpi=200, bbox_inches='tight')
    print("✅ 已保存: fig_few_shot.svg/png")
    plt.show()


def plot_per_class_heatmap(res_noisy):
    """图4：噪声条件下各方法各类别识别率热力图"""
    data_mat = np.array([res_noisy[m][2] * 100 for m in METHOD_LIST])  # (5, 6)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(data_mat, aspect='auto', cmap='RdYlGn', vmin=60, vmax=100)

    ax.set_xticks(range(len(CLASS_NAMES))); ax.set_xticklabels(CLASS_NAMES, fontsize=12)
    ax.set_yticks(range(len(METHOD_LIST))); ax.set_yticklabels(METHOD_LIST, fontsize=11)

    for i in range(len(METHOD_LIST)):
        for j in range(len(CLASS_NAMES)):
            val = data_mat[i, j]
            color = 'white' if val < 75 else 'black'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('识别准确率（%）', fontsize=11)
    ax.set_title('含测量噪声条件下各方法各类别识别准确率热力图', fontsize=13, pad=10)
    plt.tight_layout()
    plt.savefig(figure_path('fig_per_class_heatmap.svg'), format='svg', bbox_inches='tight')
    plt.savefig(figure_path('fig_per_class_heatmap.png'), dpi=200, bbox_inches='tight')
    print("✅ 已保存: fig_per_class_heatmap.svg/png")
    plt.show()


def plot_radar(res_clean, res_noisy, few_shot_results, timings):
    """图5：综合能力雷达图"""
    # 5个维度：理想准确率 / 噪声准确率 / 小样本准确率(20/类) / 可解释性(主观) / 训练效率
    labels = ['理想准确率', '噪声鲁棒性', '小样本性能', '可解释性', '训练效率']
    N = len(labels)

    # 手动打分(0~1)，理想准确率和噪声准确率来自实验，其余相对赋值
    interpret_score = {'KNN': 0.55, 'SVM': 0.35, '纯RF': 0.60, 'MLP': 0.25, 'PINN-RF(本文)': 0.98}
    max_time = max(timings.values()) if timings else 1.0

    scores = {}
    for m in METHOD_LIST:
        acc_c  = res_clean[m][0]
        acc_n  = res_noisy[m][0]
        fs_val = (few_shot_results[m][0] if few_shot_results[m] else 50) / 100.0
        interp = interpret_score[m]
        eff    = 1.0 - (timings.get(m, 0) / (max_time + 1e-6)) * 0.5
        scores[m] = [acc_c, acc_n, fs_val, interp, eff]

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=9)
    ax.grid(color='grey', linestyle='--', linewidth=0.6, alpha=0.5)

    for m in METHOD_LIST:
        vals = scores[m] + scores[m][:1]
        lw = 2.8 if m == 'PINN-RF(本文)' else 1.5
        ls = '-'
        ax.plot(angles, vals, color=COLORS[m], linewidth=lw, linestyle=ls, label=m, zorder=3)
        ax.fill(angles, vals, color=COLORS[m], alpha=0.08 if m != 'PINN-RF(本文)' else 0.18)

    ax.set_title('各方法综合能力对比雷达图', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.32, 1.18), fontsize=11)
    plt.tight_layout()
    plt.savefig(figure_path('fig_radar.svg'), format='svg', bbox_inches='tight')
    plt.savefig(figure_path('fig_radar.png'), dpi=200, bbox_inches='tight')
    print("✅ 已保存: fig_radar.svg/png")
    plt.show()


def plot_pinn_confusion(res_noisy):
    """图6：本文方法PINN-RF混淆矩阵（噪声条件）"""
    cm  = res_noisy['PINN-RF(本文)'][3]
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    acc = res_noisy['PINN-RF(本文)'][0]

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sns.heatmap(cm_pct, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                square=True, linewidths=0.4, linecolor='white',
                cbar_kws={'label': '识别准确率', 'shrink': 0.82}, ax=ax)
    ax.set_title(f'PINN-RF 故障诊断混淆矩阵（含噪声，总准确率: {acc*100:.2f}%）',
                 fontsize=13, pad=10)
    ax.set_xlabel('预测故障类型', fontsize=13)
    ax.set_ylabel('真实故障类型', fontsize=13)
    ax.tick_params(axis='x', rotation=30, labelsize=11)
    ax.tick_params(axis='y', rotation=0, labelsize=11)
    plt.tight_layout()
    plt.savefig(figure_path('fig_pinn_confusion_noisy.svg'), format='svg', bbox_inches='tight')
    plt.savefig(figure_path('fig_pinn_confusion_noisy.png'), dpi=200, bbox_inches='tight')
    print("✅ 已保存: fig_pinn_confusion_noisy.svg/png")
    plt.show()


# ═══════════════════════════════════════════════════════════════════
#  打印汇总表
# ═══════════════════════════════════════════════════════════════════

def print_summary_table(res_clean, res_noisy, timings):
    sep = "=" * 90
    print(f"\n{sep}")
    print("  表 X  各方法性能对比汇总（理想条件 / 含噪声条件）")
    print(sep)
    header = f"{'方法':<16}{'理想Acc':>9}{'理想F1':>9}{'噪声Acc':>9}{'噪声F1':>9}{'下降(pp)':>10}{'训练(s)':>10}"
    print(header); print("-" * 90)
    for m in METHOD_LIST:
        ac, f1c = res_clean[m][0]*100, res_clean[m][1]*100
        an, f1n = res_noisy[m][0]*100, res_noisy[m][1]*100
        drop = ac - an
        t    = timings.get(m, '-')
        print(f"{m:<16}{ac:>8.2f}%{f1c:>8.2f}%{an:>8.2f}%{f1n:>8.2f}%{drop:>9.2f}pp{str(t):>10}")
    print(sep)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)

    print(">>> 加载数据...")
    X_raw, Y = load_raw_features()

    print(">>> PINN提取物理特征（理想条件）...")
    X_phys = extract_pinn_features(X_raw)

    print(">>> 生成含噪声数据（σ=5%）...")
    X_raw_noisy = add_measurement_noise(X_raw, noise_level=0.05, missing_rate=0.03)
    X_phys_noisy = extract_pinn_features(X_raw_noisy)

    # 实验1：准确率与鲁棒性
    print("\n>>> 实验1：准确率与噪声鲁棒性对比...")
    res_clean, res_noisy, timings = run_accuracy_experiment(
        X_raw, X_raw_noisy, X_phys, X_phys_noisy, Y)

    # 实验2：小样本
    few_shot_results, sample_sizes = run_few_shot_experiment(X_raw, X_phys, Y,
                                                              sample_sizes=(20, 50, 100, 150, 200))

    # 汇总表
    print_summary_table(res_clean, res_noisy, timings)

    # 绘图
    print("\n>>> 生成对比图表...")
    plot_dual_accuracy(res_clean, res_noisy)
    plot_robustness_drop(res_clean, res_noisy)
    plot_few_shot(few_shot_results, sample_sizes)
    plot_per_class_heatmap(res_noisy)
    plot_radar(res_clean, res_noisy, few_shot_results, timings)
    plot_pinn_confusion(res_noisy)

    print("\n🎉 全部实验完成！共生成 6 张图 + 1 张汇总表。")
