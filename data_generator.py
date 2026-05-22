"""
data_generator_v2.py  ——  修复版数据生成器
核心修复：收紧热斑与局部阴影的参数范围，使二者在物理参数空间中更加可分
  局部阴影：Iph = 30%~55%（遮挡较深），Rsh 正常
  热 斑  ：Iph = 55%~70%（遮挡较浅，但漏电极严重），Rsh = 0.5~2.5Ω（比原来更低更窄）
这样热斑的 Rsh 特征更突出，PINN 更容易区分二者。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('./data', exist_ok=True)


class PVDataGeneratorV2:
    def __init__(self):
        self.Iph_0  = 15.60
        self.I0_0   = 0.03e-9
        self.n_0    = 1.082
        self.Rs_0   = 0.19
        self.Rsh_0  = 500.0
        self.Vt_cell = 0.026
        self.Ns_total = 72

    def _solve_voltage_brentq(self, I_target, params, num_cells):
        Iph, I0, n, Rs, Rsh = params
        Vt = self.Vt_cell * num_cells * n
        I0_bypass  = 1e-5
        Vt_bypass  = 0.052

        def error_func(V):
            V_j = V + I_target * Rs
            try:
                term_exp = np.exp(V_j / Vt)
            except OverflowError:
                term_exp = 1e10
            I_diode = I0 * (term_exp - 1)
            I_leak  = V_j / Rsh
            I_pv    = Iph - I_diode - I_leak
            I_bypass = I0_bypass * (np.exp(-V / Vt_bypass) - 1) if V < -0.5 else 0
            return (I_pv + I_bypass) - I_target

        try:
            return brentq(error_func, -25, 70)
        except:
            return 0.0

    def generate_random_sample(self, fault_type, num_points=100):
        Iph    = self.Iph_0  * np.random.uniform(0.98, 1.02)
        Rs_base = self.Rs_0  * np.random.uniform(0.98, 1.02)
        Rsh_base = self.Rsh_0 * np.random.uniform(0.95, 1.05)
        I0_base  = self.I0_0  * np.random.uniform(0.9, 1.1)
        params   = [Iph, I0_base, self.n_0, Rs_base, Rsh_base]
        label    = 0

        cells_per_sub = 24

        if fault_type == 'normal':
            label = 0

        elif fault_type == 'aging':
            label = 1
            params[3] = Rs_base * np.random.uniform(3.0, 8.0)

        elif fault_type == 'pid':
            label = 2
            params[4] = Rsh_base * np.random.uniform(0.02, 0.10)
            params[1] = I0_base * np.random.uniform(10, 100)

        elif fault_type == 'diode_short':
            label = 3

        elif fault_type == 'partial_shading':
            label = 4
            # ★ 修复：遮挡程度收窄到 30%~55%（更深遮挡，Iph较低，与热斑拉开差距）
            shading_factor = np.random.uniform(0.30, 0.55)
            params_bad = list(params)
            params_bad[0] = Iph * shading_factor

        elif fault_type == 'hot_spot':
            label = 5
            # ★ 修复：Iph 55%~70%（较浅遮挡），Rsh 极低 0.5~2.5Ω（漏电更严重）
            shading_factor = np.random.uniform(0.55, 0.70)
            params_bad = list(params)
            params_bad[0] = Iph * shading_factor
            params_bad[4] = np.random.uniform(0.5, 2.5)   # 比原来更低更窄

        I_scan = np.linspace(0, Iph + 0.5, num_points)
        V_scan = []

        for I in I_scan:
            if fault_type in ['partial_shading', 'hot_spot']:
                V_good = self._solve_voltage_brentq(I, params,      cells_per_sub * 2)
                V_bad  = self._solve_voltage_brentq(I, params_bad,  cells_per_sub)
                V = V_good + V_bad
            elif fault_type == 'diode_short':
                V = self._solve_voltage_brentq(I, params, cells_per_sub * 2)
            else:
                V = self._solve_voltage_brentq(I, params, self.Ns_total)

            V_scan.append(max(0, V))

        return np.array(V_scan), I_scan, label, np.array(params)

    def create_dataset(self, samples_per_class=200):
        print("生成修复版数据集 (v2)...")
        all_V, all_I, all_labels, all_params = [], [], [], []
        fault_types = ['normal', 'aging', 'pid', 'diode_short', 'partial_shading', 'hot_spot']

        for f_type in fault_types:
            print(f"  生成: {f_type} ...")
            for _ in range(samples_per_class):
                V, I, label, p = self.generate_random_sample(f_type)
                all_V.append(V); all_I.append(I)
                all_labels.append(label); all_params.append(p)

        X_V      = np.array(all_V)
        X_I      = np.array(all_I)
        Y_labels = np.array(all_labels)
        Y_params = np.array(all_params)

        np.savez('./data/train_dataset.npz',
                 V=X_V, I=X_I, label=Y_labels, params=Y_params)
        print(f"✅ 数据集已保存，样本数: {len(X_V)}")

        # 验证图：对比局部阴影与热斑曲线
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        idx_shading = np.where(Y_labels == 4)[0][:3]
        idx_hotspot = np.where(Y_labels == 5)[0][:3]
        for i in idx_shading:
            axes[0].plot(X_V[i], X_I[i], 'b-', alpha=0.6)
        for i in idx_hotspot:
            axes[1].plot(X_V[i], X_I[i], 'r-', alpha=0.6)
        axes[0].set_title('局部阴影 (Iph=30~55%)')
        axes[1].set_title('热斑 (Iph=55~70%, Rsh=0.5~2.5Ω)')
        for ax in axes:
            ax.set_xlabel('Voltage (V)'); ax.set_ylabel('Current (A)'); ax.grid(True)
        plt.tight_layout(); plt.savefig('fault_curves_v2.png', dpi=150); plt.show()

        return X_V, Y_labels


if __name__ == "__main__":
    gen = PVDataGeneratorV2()
    gen.create_dataset(samples_per_class=200)