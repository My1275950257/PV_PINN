import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn


class PV_PINN(nn.Module):
    # === 修改 1: input_dim 默认为 200 (100个电压 + 100个电流) ===
    def __init__(self, input_dim=200):
        super(PV_PINN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh()
        )

        self.head = nn.Linear(32, 5)
        self.softplus = nn.Softplus()

    def forward(self, x):
        features = self.encoder(x)
        raw_output = self.head(features)

        # 参数反归一化 (保持不变)
        pred_Iph = self.softplus(raw_output[:, 0]) * 15.0  # 适配 15A
        pred_I0 = torch.pow(10, -5 - self.softplus(raw_output[:, 1]))
        pred_n = 1.0 + torch.sigmoid(raw_output[:, 2])
        pred_Rs = self.softplus(raw_output[:, 3])
        pred_Rsh = 10.0 + self.softplus(raw_output[:, 4]) * 500.0

        return torch.stack([pred_Iph, pred_I0, pred_n, pred_Rs, pred_Rsh], dim=1)

    # calculate_physics_loss 保持不变
    def calculate_physics_loss(self, V_tensor, I_tensor, params_tensor):
        Iph = params_tensor[:, 0].unsqueeze(1)
        I0 = params_tensor[:, 1].unsqueeze(1)
        n = params_tensor[:, 2].unsqueeze(1)
        Rs = params_tensor[:, 3].unsqueeze(1)
        Rsh = params_tensor[:, 4].unsqueeze(1)

        Vt_cell = 0.026
        Ns = 72  # 你的组件是72片
        Vt = Ns * Vt_cell * n

        # 为了数值稳定性，加个钳位
        V_j = V_tensor + I_tensor * Rs
        term_exp = torch.exp(torch.clamp(V_j / Vt, max=50)) - 1
        term_leak = V_j / Rsh

        I_theory = Iph - I0 * term_exp - term_leak
        loss_f = torch.mean((I_tensor - I_theory) ** 2)
        return loss_f