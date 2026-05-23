import numpy as np


EPS = 1e-8


def _row_features(v_row, i_row):
    """Extract compact electrical descriptors from one normalized I-V curve."""
    v = np.asarray(v_row, dtype=np.float64)
    i = np.asarray(i_row, dtype=np.float64)

    order = np.argsort(v)
    v = v[order]
    i = i[order]

    p = v * i
    idx_mpp = int(np.argmax(p))
    voc = float(np.max(v))
    isc = float(np.max(i))
    vmp = float(v[idx_mpp])
    imp = float(i[idx_mpp])
    pmax = float(p[idx_mpp])
    fill_factor = pmax / (voc * isc + EPS)

    dv = np.diff(v)
    di = np.diff(i)
    slope = di / (dv + EPS)
    slope_mean = float(np.mean(slope))
    slope_std = float(np.std(slope))
    slope_min = float(np.min(slope))
    slope_max = float(np.max(slope))

    curvature = np.diff(slope)
    curve_abs_mean = float(np.mean(np.abs(curvature))) if curvature.size else 0.0
    curve_abs_max = float(np.max(np.abs(curvature))) if curvature.size else 0.0

    # Mismatch and bypass-diode events usually show up as abrupt slope changes.
    jump_threshold = np.percentile(np.abs(curvature), 90) if curvature.size else 0.0
    jump_count = float(np.sum(np.abs(curvature) > jump_threshold)) if jump_threshold > 0 else 0.0

    left = slice(0, max(3, len(v) // 5))
    right = slice(max(0, len(v) - max(3, len(v) // 5)), len(v))
    low_voltage_slope = float(np.mean(slope[left])) if slope.size else 0.0
    high_voltage_slope = float(np.mean(slope[right.start - 1:])) if slope.size else 0.0

    return [
        voc, isc, vmp, imp, pmax, fill_factor,
        float(np.mean(v)), float(np.std(v)), float(np.mean(i)), float(np.std(i)),
        slope_mean, slope_std, slope_min, slope_max,
        curve_abs_mean, curve_abs_max, jump_count,
        low_voltage_slope, high_voltage_slope,
    ]


def extract_iv_curve_features(x_raw):
    """
    Extract literature-style I-V electrical features from normalized raw input.

    x_raw is expected to be [V_norm(100), I_norm(100)] per row, matching the
    project data pipeline. The returned matrix is intentionally compact so it
    can be concatenated with PINN-inverted physical parameters.
    """
    x_raw = np.asarray(x_raw, dtype=np.float64)
    if x_raw.ndim != 2 or x_raw.shape[1] % 2 != 0:
        raise ValueError("x_raw must be a 2D array with voltage and current halves")

    half = x_raw.shape[1] // 2
    v_mat = x_raw[:, :half]
    i_mat = x_raw[:, half:]
    features = [_row_features(v_mat[k], i_mat[k]) for k in range(x_raw.shape[0])]
    return np.asarray(features, dtype=np.float32)
