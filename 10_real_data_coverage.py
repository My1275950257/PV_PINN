import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_utils import load_raw_features
from paths import PROJECT_ROOT, figure_path, table_path
from plot_utils import setup_chinese_font


REAL_DATA_PATH = PROJECT_ROOT / "data" / "real_operation_points.csv"


def _load_real_points(path=REAL_DATA_PATH):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    lower_cols = {c.lower(): c for c in df.columns}
    voltage_col = lower_cols.get("voltage") or lower_cols.get("v")
    current_col = lower_cols.get("current") or lower_cols.get("i")
    if voltage_col is None or current_col is None:
        raise ValueError("real_operation_points.csv must contain voltage/current or V/I columns")
    return df[[voltage_col, current_col]].rename(columns={voltage_col: "voltage", current_col: "current"})


def run_real_data_coverage():
    setup_chinese_font()
    x_raw, _ = load_raw_features()
    half = x_raw.shape[1] // 2
    sim_v = (x_raw[:, :half] * 60.0).reshape(-1)
    sim_i = (x_raw[:, half:] * 16.0).reshape(-1)

    sim_summary = pd.DataFrame([{
        "source": "simulation",
        "voltage_min": sim_v.min(),
        "voltage_max": sim_v.max(),
        "voltage_mean": sim_v.mean(),
        "current_min": sim_i.min(),
        "current_max": sim_i.max(),
        "current_mean": sim_i.mean(),
        "point_count": len(sim_v),
    }])

    real_df = _load_real_points()
    summaries = [sim_summary]

    plt.figure(figsize=(7, 5))
    plt.scatter(sim_v, sim_i, s=2, alpha=0.12, label="仿真样本空间", color="#4C78A8")

    if real_df is not None:
        rv = real_df["voltage"].to_numpy(dtype=float)
        ri = real_df["current"].to_numpy(dtype=float)
        summaries.append(pd.DataFrame([{
            "source": "real_operation",
            "voltage_min": rv.min(),
            "voltage_max": rv.max(),
            "voltage_mean": rv.mean(),
            "current_min": ri.min(),
            "current_max": ri.max(),
            "current_mean": ri.mean(),
            "point_count": len(rv),
        }]))
        plt.scatter(rv, ri, s=14, alpha=0.65, label="真实运行点", color="#E45756")
    else:
        template = PROJECT_ROOT / "data" / "real_operation_points_template.csv"
        pd.DataFrame({
            "voltage": [],
            "current": [],
            "irradiance": [],
            "temperature": [],
        }).to_csv(template, index=False, encoding="utf-8-sig")
        print(f"⚠️ 未找到真实数据: {REAL_DATA_PATH}")
        print(f"   已生成模板: {template}")
        print("   填入 voltage/current 列后重新运行本脚本即可叠加真实运行点。")

    summary = pd.concat(summaries, ignore_index=True)
    out_csv = table_path("real_data_coverage_summary.csv")
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")

    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.title("仿真样本空间与真实运行点覆盖性分析")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path("fig_real_data_coverage.png"), dpi=200, bbox_inches="tight")
    plt.show()

    print(f"✅ 覆盖性统计已保存: {out_csv}")


if __name__ == "__main__":
    run_real_data_coverage()
