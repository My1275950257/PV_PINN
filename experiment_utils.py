import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

from iv_features import extract_iv_curve_features
from pinn_model import PV_PINN


DATA_PATH = "./data/train_dataset.npz"
MODEL_PATH = "pinn_model_v3.pth"
RANDOM_SEED = 42
CLASS_NAMES = ["正常", "老化", "PID衰减", "二极管短路", "局部阴影", "热斑"]
PINN_FEATURE_NAMES = ["Iph", "I0", "n", "Rs", "Rsh"]
IV_FEATURE_NAMES = [
    "Voc", "Isc", "Vmp", "Imp", "Pmax", "FF",
    "V_mean", "V_std", "I_mean", "I_std",
    "slope_mean", "slope_std", "slope_min", "slope_max",
    "curvature_mean_abs", "curvature_max_abs", "jump_count",
    "low_voltage_slope", "high_voltage_slope",
]
FUSED_FEATURE_NAMES = PINN_FEATURE_NAMES + IV_FEATURE_NAMES


def load_raw_features(data_path=DATA_PATH):
    data = np.load(data_path)
    x_v = data["V"] / 60.0
    x_i = data["I"] / 16.0
    x_raw = np.concatenate([x_v, x_i], axis=1).astype(np.float32)
    y = data["label"].astype(int)
    return x_raw, y


def add_measurement_noise(x_raw, noise_level=0.05, missing_rate=0.03, random_seed=RANDOM_SEED):
    rng = np.random.default_rng(random_seed)
    x_noisy = x_raw.copy()
    noise_std = noise_level * np.max(np.abs(x_raw), axis=1, keepdims=True)
    x_noisy += rng.normal(0, 1, x_noisy.shape) * noise_std
    mask = rng.random(x_noisy.shape) < missing_rate
    x_noisy[mask] *= rng.uniform(0.7, 1.3, mask.sum())
    return np.clip(x_noisy, -0.1, 1.1).astype(np.float32)


def load_pinn_model(model_path=MODEL_PATH, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PV_PINN(input_dim=200).to(device)
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device


def extract_pinn_features(x_raw, model_path=MODEL_PATH, batch_size=512):
    model, device = load_pinn_model(model_path)
    x_tensor = torch.tensor(x_raw, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=False)
    feats = []
    with torch.no_grad():
        for (batch,) in loader:
            feats.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


def build_feature_sets(x_raw, model_path=MODEL_PATH):
    x_pinn = extract_pinn_features(x_raw, model_path=model_path)
    x_iv = extract_iv_curve_features(x_raw)
    x_fused = np.concatenate([x_pinn, x_iv], axis=1).astype(np.float32)
    return {
        "raw_iv_curve": x_raw,
        "iv_features": x_iv,
        "pinn_params": x_pinn,
        "pinn_iv_fused": x_fused,
    }


def make_classifier(method, random_seed=RANDOM_SEED):
    if method == "KNN":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7, metric="euclidean", n_jobs=-1)),
        ])
    if method == "SVM":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale", random_state=random_seed)),
        ])
    if method == "RF":
        return RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=random_seed, n_jobs=-1
        )
    if method == "MLP":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                max_iter=500,
                random_state=random_seed,
                early_stopping=True,
                validation_fraction=0.1,
            )),
        ])
    if method == "PINN-RF":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=8, random_state=random_seed, n_jobs=-1
            )),
        ])
    raise ValueError(f"Unknown method: {method}")


def evaluate_classifier(method, x, y, test_size=0.2, random_seed=RANDOM_SEED):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    clf = make_classifier(method, random_seed=random_seed)
    t0 = time.time()
    clf.fit(x_train, y_train)
    train_time = time.time() - t0
    y_pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "confusion_matrix": cm,
        "per_class_accuracy": cm.diagonal() / cm.sum(axis=1),
        "train_time": train_time,
        "model": clf,
    }
