"""
train_v2.py  ——  修复版训练脚本
核心改动：
1. 增强热斑 Rsh 的损失权重（weight=50，原来30）
2. 降低热斑阈值检测边界（<2.5Ω 触发最高惩罚）
3. 物理损失权重从 0.2 调整到 0.5，增强物理约束正则化
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from pinn_model import PV_PINN
from paths import figure_path

BATCH_SIZE = 128
LR         = 0.003
EPOCHS     = 1200   # 多训一些


def load_data():
    data  = np.load('./data/train_dataset.npz')
    X_V   = torch.tensor(data['V'], dtype=torch.float32)
    X_I   = torch.tensor(data['I'], dtype=torch.float32)
    Y_params = torch.tensor(data['params'], dtype=torch.float32)
    V_norm = X_V / 60.0
    I_norm = X_I / 16.0
    X_input = torch.cat([V_norm, I_norm], dim=1)
    return DataLoader(TensorDataset(X_input, X_V, X_I, Y_params),
                      batch_size=BATCH_SIZE, shuffle=True)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    dataloader = load_data()
    model = PV_PINN(input_dim=200).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    loss_history = []

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_input, batch_V, batch_I, batch_params_true in dataloader:
            batch_input      = batch_input.to(device)
            batch_V          = batch_V.to(device)
            batch_I          = batch_I.to(device)
            batch_params_true = batch_params_true.to(device)

            pred_params = model(batch_input)

            # ── Rs 加权损失（老化检测）
            mse_rs    = (pred_params[:, 3] - batch_params_true[:, 3]) ** 2
            w_rs      = torch.ones_like(batch_params_true[:, 3])
            w_rs[batch_params_true[:, 3] > 0.25] = 5.0
            w_rs[batch_params_true[:, 3] > 0.5] = 20.0
            loss_rs   = torch.mean(w_rs * mse_rs) * 100.0

            # ── Rsh 加权损失（热斑+PID检测）★ 增强热斑权重
            mse_rsh   = (pred_params[:, 4] / 100.0 - batch_params_true[:, 4] / 100.0) ** 2
            w_rsh     = torch.ones_like(batch_params_true[:, 4])
            w_rsh[batch_params_true[:, 4] < 250.0] = 4.0
            w_rsh[batch_params_true[:, 4] < 50.0]  = 10.0
            w_rsh[batch_params_true[:, 4] < 30.0]  = 20.0
            w_rsh[batch_params_true[:, 4] < 5.0]   = 35.0
            w_rsh[batch_params_true[:, 4] < 2.5]   = 50.0
            loss_rsh  = torch.mean(w_rsh * mse_rsh) * 15.0

            # ── 物理残差损失（★ 权重0.2→0.5，增强物理约束）
            loss_phys = model.calculate_physics_loss(batch_V, batch_I, pred_params) * 0.5

            loss = loss_rs + loss_rsh + loss_phys
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg = total_loss / len(dataloader)
        loss_history.append(avg)

        if epoch % 100 == 0:
            rs_p  = pred_params[0, 3].item()
            rs_t  = batch_params_true[0, 3].item()
            rsh_p = pred_params[0, 4].item()
            rsh_t = batch_params_true[0, 4].item()
            print(f"Epoch {epoch:4d} | Loss={avg:.4f} | "
                  f"Rs: pred={rs_p:.4f} true={rs_t:.4f} | "
                  f"Rsh: pred={rsh_p:.2f} true={rsh_t:.2f}")

    torch.save(model.state_dict(), 'pinn_model_v3.pth')
    print("✅ 训练完成，已保存 pinn_model_v3.pth")

    plt.figure(figsize=(8, 4))
    plt.plot(loss_history)
    plt.yscale('log')
    plt.title("Training Loss (v2)")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.grid(True); plt.tight_layout()
    plt.savefig(figure_path('training_loss_v2.png'), dpi=150)
    plt.show()


if __name__ == "__main__":
    train()
