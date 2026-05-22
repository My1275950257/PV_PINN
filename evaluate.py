import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from pinn_model import PV_PINN

# 1. 设置环境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 加载数据 (和 train.py 一样)
data = np.load('./data/train_dataset.npz')
X_V = torch.tensor(data['V'], dtype=torch.float32)
X_I = torch.tensor(data['I'], dtype=torch.float32)
Y_params = torch.tensor(data['params'], dtype=torch.float32)

# 归一化并拼接
V_norm = X_V / 60.0
I_norm = X_I / 16.0
X_input = torch.cat([V_norm, I_norm], dim=1)

dataset = TensorDataset(X_input, Y_params)
dataloader = DataLoader(dataset, batch_size=200, shuffle=False)

# 3. 加载模型
model = PV_PINN(input_dim=200).to(device)
# 注意：这里加载你刚刚训练完保存的 v3.pth (或者 v4, 取决于你上次保存的名字)
model.load_state_dict(torch.load('pinn_model_v3.pth', map_location=device))
model.eval()

# 4. 推理
true_rs, pred_rs = [], []
true_rsh, pred_rsh = [], []

with torch.no_grad():
    for batch_input, batch_params in dataloader:
        batch_input = batch_input.to(device)
        p = model(batch_input)

        true_rs.extend(batch_params[:, 3].numpy())
        pred_rs.extend(p[:, 3].cpu().numpy())

        true_rsh.extend(batch_params[:, 4].numpy())
        pred_rsh.extend(p[:, 4].cpu().numpy())

# 5. 绘图 (单独生成两张图)
# --- 全局基础设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ================= 图 1: Rs 预测效果 =================
plt.figure(figsize=(7, 6)) # 创建第一张独立的图

plt.scatter(true_rs, pred_rs, alpha=0.6, c='blue', s=15, label='预测样本')
lims = [0, max(max(true_rs), max(pred_rs)) * 1.05]
plt.plot(lims, lims, 'r--', linewidth=2, label='理想对角线')

# [字体调节重点] 局部调整各项字体大小
plt.title('串联电阻 反演精度', fontsize=16)            # 调节标题字体
plt.xlabel(r'真实值 ($\Omega$)', fontsize=14)         # 调节X轴标签字体
plt.ylabel(r'PINN 预测值 ($\Omega$)', fontsize=14)    # 调节Y轴标签字体
plt.xticks(fontsize=12)                               # 调节X轴坐标刻度字体
plt.yticks(fontsize=12)                               # 调节Y轴坐标刻度字体
plt.legend(fontsize=12)                               # 调节图例字体

plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# 保存为 SVG 矢量图 (推荐用于 Word/PPT)，bbox_inches='tight' 可去除多余白边
plt.savefig('Rs_prediction.svg', format='svg', bbox_inches='tight')
plt.show() # 显示第一张图 (关闭窗口后会继续画第二张)

# ================= 图 2: Rsh 预测效果 =================
plt.figure(figsize=(7, 6)) # 创建第二张独立的图

plt.scatter(true_rsh, pred_rsh, alpha=0.6, c='green', s=15, label='预测样本')
lims_rsh = [0, max(max(true_rsh), max(pred_rsh)) * 1.05]
plt.plot(lims_rsh, lims_rsh, 'r--', linewidth=2, label='理想对角线')

# [字体调节重点] 保持与第一张图一致的字体大小
plt.title('并联电阻 反演精度', fontsize=16)
plt.xlabel(r'真实值 ($\Omega$)', fontsize=14)
plt.ylabel(r'PINN 预测值 ($\Omega$)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)

plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# 保存为 SVG 矢量图
plt.savefig('Rsh_prediction.svg', format='svg', bbox_inches='tight')
plt.show()