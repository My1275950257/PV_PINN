import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier  # 优化点 1：引入随机森林
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# 引入你的模型定义
from pinn_model import PV_PINN

# ================= 配置 =================
MODEL_PATH = 'pinn_model_v3.pth'  # 请确保文件名与你保存的一致
DATA_PATH = './data/train_dataset.npz'

# 汉化标签列表
CLASS_NAMES_CN = ['正常', '老化', 'PID衰减', '二极管短路', '局部阴影', '热斑']

# =======================================

def plot_confusion_matrix_cn():
    # 1. 准备环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 正在使用设备: {device}")

    # 2. 加载数据
    print("正在加载数据集...")
    data = np.load(DATA_PATH)
    X_V = torch.tensor(data['V'], dtype=torch.float32)
    X_I = torch.tensor(data['I'], dtype=torch.float32)
    Y_labels = data['label']

    # 归一化并拼接
    V_norm = X_V / 60.0
    I_norm = X_I / 16.0
    X_input = torch.cat([V_norm, I_norm], dim=1)

    # 3. 加载训练好的 PINN 模型
    print("正在加载 PINN 模型...")
    model = PV_PINN(input_dim=200).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 4. 使用 PINN 提取物理参数 (特征提取)
    print("正在使用 PINN 反演物理参数...")
    pred_params_list = []

    batch_size = 500
    dataset = TensorDataset(X_input)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch_in in dataloader:
            batch_in = batch_in[0].to(device)
            preds = model(batch_in)
            pred_params_list.append(preds.cpu().numpy())

    X_features = np.concatenate(pred_params_list, axis=0)

    # 5. 使用随机森林验证参数可分性
    print("正在验证参数区分度 (Random Forest)...")

    X_train, X_test, y_train, y_test = train_test_split(
        X_features, Y_labels, test_size=0.2, random_state=42
    )

    # 优化点 2：使用随机森林分类器，n_estimators=100 意味着集成 100 棵树
    classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = classifier.score(X_test, y_test)
    print(f"✅ 基于 PINN 参数的诊断准确率: {accuracy * 100:.2f}%")

    # 6. 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 7. 绘制中文热力图
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10, 8), dpi=120)

    sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=CLASS_NAMES_CN, yticklabels=CLASS_NAMES_CN,
                square=True, cbar_kws={'label': '识别准确率 '})

    plt.title(f'总体准确率: {accuracy * 100:.2f}%', fontsize=18)
    plt.xlabel('预测故障类型', fontsize=20)
    plt.ylabel('真实故障类型', fontsize=20)

    plt.xticks(rotation=45, fontsize=18)
    plt.yticks(rotation=0, fontsize=18)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_confusion_matrix_cn()