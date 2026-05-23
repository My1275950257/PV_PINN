"""
01_generate_data.py

Generate synthetic I-V datasets for PV fault diagnosis.

Two modes are provided:
- easy: keeps the original clean class boundaries for quick debugging.
- paper: adds irradiance/temperature variation, mild faults, boundary samples,
  measurement noise, and partial overlap between shading and hot-spot faults.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from paths import figure_path

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
os.makedirs("./data", exist_ok=True)


FAULT_TYPES = ["normal", "aging", "pid", "diode_short", "partial_shading", "hot_spot"]
FAULT_LABELS = {name: idx for idx, name in enumerate(FAULT_TYPES)}


class PVDataGenerator:
    def __init__(self, difficulty="paper", seed=42):
        self.difficulty = difficulty
        self.rng = np.random.default_rng(seed)

        self.Iph_0 = 15.60
        self.I0_0 = 0.03e-9
        self.n_0 = 1.082
        self.Rs_0 = 0.19
        self.Rsh_0 = 500.0
        self.Vt_cell_25c = 0.026
        self.Ns_total = 72

    def _sample_environment(self):
        if self.difficulty == "easy":
            return 1000.0, 25.0
        irradiance = self.rng.uniform(200.0, 1000.0)
        temperature = self.rng.uniform(10.0, 70.0)
        return irradiance, temperature

    def _base_params(self, irradiance, temperature):
        temp_k = temperature + 273.15
        ref_k = 298.15
        vt_cell = self.Vt_cell_25c * temp_k / ref_k

        iph = self.Iph_0 * (irradiance / 1000.0) * (1.0 + 0.0005 * (temperature - 25.0))
        iph *= self.rng.uniform(0.98, 1.02)
        iph = max(0.2, iph)

        rs = self.Rs_0 * self.rng.uniform(0.98, 1.02)
        rsh = self.Rsh_0 * self.rng.uniform(0.95, 1.05)
        i0 = self.I0_0 * (temp_k / ref_k) ** 3 * self.rng.uniform(0.9, 1.1)
        n = self.n_0 * self.rng.uniform(0.98, 1.02)
        return [iph, i0, n, rs, rsh], vt_cell

    def _solve_voltage_brentq(self, i_target, params, num_cells, vt_cell):
        iph, i0, n, rs, rsh = params
        vt = vt_cell * num_cells * n
        i0_bypass = 1e-5
        vt_bypass = 0.052

        def error_func(v):
            v_j = v + i_target * rs
            term_exp = np.exp(np.clip(v_j / vt, -80, 80))
            i_diode = i0 * (term_exp - 1)
            i_leak = v_j / max(rsh, 0.1)
            i_pv = iph - i_diode - i_leak
            i_bypass = i0_bypass * (np.exp(np.clip(-v / vt_bypass, -80, 80)) - 1) if v < -0.5 else 0
            return (i_pv + i_bypass) - i_target

        try:
            return brentq(error_func, -25, 80, maxiter=100)
        except ValueError:
            return 0.0

    def _apply_measurement_noise(self, v_scan, i_scan):
        if self.difficulty == "easy":
            return v_scan, i_scan

        v = v_scan.copy()
        i = i_scan.copy()
        v_noise = self.rng.uniform(0.005, 0.02)
        i_noise = self.rng.uniform(0.005, 0.02)
        v += self.rng.normal(0, v_noise * max(np.max(v), 1.0), size=v.shape)
        i += self.rng.normal(0, i_noise * max(np.max(i), 1.0), size=i.shape)

        spike_rate = self.rng.uniform(0.01, 0.03)
        spike_mask = self.rng.random(v.shape) < spike_rate
        if np.any(spike_mask):
            v[spike_mask] *= self.rng.uniform(0.96, 1.04, spike_mask.sum())
            i[spike_mask] *= self.rng.uniform(0.94, 1.06, spike_mask.sum())

        return np.clip(v, 0, None), np.clip(i, 0, None)

    def generate_random_sample(self, fault_type, num_points=100):
        irradiance, temperature = self._sample_environment()
        params, vt_cell = self._base_params(irradiance, temperature)
        iph, i0_base, _, rs_base, rsh_base = params
        label = FAULT_LABELS[fault_type]
        cells_per_sub = 24
        params_bad = None
        diagnostic_params = list(params)

        if fault_type == "normal":
            pass

        elif fault_type == "aging":
            if self.difficulty == "easy":
                rs_factor = self.rng.uniform(3.0, 8.0)
            else:
                rs_factor = self.rng.choice([
                    self.rng.uniform(1.2, 2.0),
                    self.rng.uniform(2.0, 4.0),
                    self.rng.uniform(4.0, 8.0),
                ], p=[0.35, 0.35, 0.30])
            params[3] = rs_base * rs_factor
            diagnostic_params = list(params)

        elif fault_type == "pid":
            if self.difficulty == "easy":
                rsh_factor = self.rng.uniform(0.02, 0.10)
                i0_factor = self.rng.uniform(10, 100)
            else:
                rsh_factor = self.rng.choice([
                    self.rng.uniform(0.50, 0.80),
                    self.rng.uniform(0.15, 0.50),
                    self.rng.uniform(0.02, 0.15),
                ], p=[0.30, 0.35, 0.35])
                i0_factor = self.rng.uniform(2, 100)
            params[4] = rsh_base * rsh_factor
            params[1] = i0_base * i0_factor
            diagnostic_params = list(params)

        elif fault_type == "diode_short":
            if self.difficulty == "paper":
                params[3] = rs_base * self.rng.uniform(0.9, 1.4)
                params[4] = rsh_base * self.rng.uniform(0.7, 1.05)
            diagnostic_params = list(params)

        elif fault_type == "partial_shading":
            shading_factor = self.rng.uniform(0.30, 0.55) if self.difficulty == "easy" else self.rng.uniform(0.30, 0.75)
            params_bad = list(params)
            params_bad[0] = iph * shading_factor
            if self.difficulty == "paper":
                params_bad[4] = self.rng.uniform(100.0, 600.0)
            diagnostic_params = list(params)
            diagnostic_params[0] = params_bad[0]
            diagnostic_params[4] = params_bad[4]

        elif fault_type == "hot_spot":
            shading_factor = self.rng.uniform(0.55, 0.70) if self.difficulty == "easy" else self.rng.uniform(0.40, 0.80)
            params_bad = list(params)
            params_bad[0] = iph * shading_factor
            params_bad[4] = self.rng.uniform(0.5, 2.5) if self.difficulty == "easy" else self.rng.uniform(0.5, 30.0)
            if self.difficulty == "paper":
                params_bad[3] = rs_base * self.rng.uniform(1.0, 2.5)
            diagnostic_params = list(params)
            diagnostic_params[0] = params_bad[0]
            diagnostic_params[3] = params_bad[3]
            diagnostic_params[4] = params_bad[4]

        i_scan = np.linspace(0, iph + 0.5, num_points)
        v_scan = []

        for current in i_scan:
            if fault_type in ["partial_shading", "hot_spot"]:
                v_good = self._solve_voltage_brentq(current, params, cells_per_sub * 2, vt_cell)
                v_bad = self._solve_voltage_brentq(current, params_bad, cells_per_sub, vt_cell)
                voltage = v_good + v_bad
            elif fault_type == "diode_short":
                voltage = self._solve_voltage_brentq(current, params, cells_per_sub * 2, vt_cell)
            else:
                voltage = self._solve_voltage_brentq(current, params, self.Ns_total, vt_cell)
            v_scan.append(max(0, voltage))

        v_scan = np.asarray(v_scan)
        i_scan = np.asarray(i_scan)
        v_scan, i_scan = self._apply_measurement_noise(v_scan, i_scan)

        return v_scan, i_scan, label, np.asarray(diagnostic_params), irradiance, temperature

    def create_dataset(self, samples_per_class=2000, num_points=100, output_path="./data/train_dataset.npz"):
        print(f"生成数据集: difficulty={self.difficulty}, 每类样本={samples_per_class}")
        all_v, all_i, all_labels, all_params, all_g, all_t = [], [], [], [], [], []

        for fault_type in FAULT_TYPES:
            print(f"  生成: {fault_type} ...")
            for _ in range(samples_per_class):
                v, i, label, params, irradiance, temperature = self.generate_random_sample(fault_type, num_points)
                all_v.append(v)
                all_i.append(i)
                all_labels.append(label)
                all_params.append(params)
                all_g.append(irradiance)
                all_t.append(temperature)

        x_v = np.asarray(all_v, dtype=np.float32)
        x_i = np.asarray(all_i, dtype=np.float32)
        y_labels = np.asarray(all_labels, dtype=np.int64)
        y_params = np.asarray(all_params, dtype=np.float32)
        irradiance = np.asarray(all_g, dtype=np.float32)
        temperature = np.asarray(all_t, dtype=np.float32)

        np.savez(
            output_path,
            V=x_v,
            I=x_i,
            label=y_labels,
            params=y_params,
            irradiance=irradiance,
            temperature=temperature,
            difficulty=self.difficulty,
        )
        print(f"✅ 数据集已保存: {output_path}, 样本数: {len(x_v)}")

        self._plot_fault_curves(x_v, x_i, y_labels)
        return x_v, y_labels

    def _plot_fault_curves(self, x_v, x_i, y_labels):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        idx_shading = np.where(y_labels == FAULT_LABELS["partial_shading"])[0][:8]
        idx_hotspot = np.where(y_labels == FAULT_LABELS["hot_spot"])[0][:8]
        for idx in idx_shading:
            axes[0].plot(x_v[idx], x_i[idx], "b-", alpha=0.45)
        for idx in idx_hotspot:
            axes[1].plot(x_v[idx], x_i[idx], "r-", alpha=0.45)
        if self.difficulty == "easy":
            axes[0].set_title("局部阴影 (Iph=30~55%)")
            axes[1].set_title("热斑 (Iph=55~70%, Rsh=0.5~2.5Ω)")
        else:
            axes[0].set_title("局部阴影 (Iph=30~75%, Rsh=100~600Ω)")
            axes[1].set_title("热斑 (Iph=40~80%, Rsh=0.5~30Ω)")
        for ax in axes:
            ax.set_xlabel("Voltage (V)")
            ax.set_ylabel("Current (A)")
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(figure_path(f"fault_curves_{self.difficulty}.png"), dpi=150)
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PV fault I-V dataset.")
    parser.add_argument("--difficulty", choices=["easy", "paper"], default="paper")
    parser.add_argument("--samples-per-class", type=int, default=2000)
    parser.add_argument("--num-points", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="./data/train_dataset.npz")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generator = PVDataGenerator(difficulty=args.difficulty, seed=args.seed)
    generator.create_dataset(
        samples_per_class=args.samples_per_class,
        num_points=args.num_points,
        output_path=args.output,
    )
