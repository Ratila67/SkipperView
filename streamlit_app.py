from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import streamlit as st
import joblib
from pathlib import Path
from PIL import Image

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


APP_NAME = "Magnetic Map — Analysis Console"
TAGLINE = "Managing the invisible — Analysis Console"


# ============================================================
# CONFIG DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class Thresholds:
    t1_presence: float = 0.4
    t3_current: float = 0.5
    t4_parallel: float = 0.5


@dataclass(frozen=True)
class RunConfig:
    thresholds: Thresholds
    device: str = "cpu"
    mock_mode: bool = True


# ============================================================
# NPZ LOADER
# ============================================================

def _normalize_key(k: str) -> str:
    return "".join(ch for ch in k.lower().strip() if ch.isalnum())


def load_npz_to_hwc4(uploaded_file: Any) -> np.ndarray:
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    with np.load(uploaded_file, allow_pickle=False) as z:
        keys = list(z.files)

        want = {"bx": None, "by": None, "bz": None, "norm": None}
        norm_map = {_normalize_key(k): k for k in keys}

        for short in want:
            if short in norm_map:
                want[short] = norm_map[short]

        if all(v is not None for v in want.values()):
            arrays = [np.asarray(z[want[k]]) for k in ("bx", "by", "bz", "norm")]
            arrays = [np.squeeze(a).astype(np.float32) for a in arrays]
            shapes = {a.shape for a in arrays}
            if len(shapes) != 1:
                raise ValueError("Shapes canaux incohérentes.")
            return np.stack(arrays, axis=-1)

        for k in keys:
            a = np.asarray(z[k])
            if a.ndim == 3 and a.shape[-1] == 4:
                return a.astype(np.float32)
            if a.ndim == 3 and a.shape[0] == 4:
                return np.moveaxis(a, 0, -1).astype(np.float32)

        raise ValueError("Format .npz non reconnu.")


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_models() -> Dict[str, Any]:
    return {
        "task1_pipeline": joblib.load(Path("models/model_task1.pkl")),
    }


# ============================================================
# PREPROCESS (IDENTIQUE TRAINING)
# ============================================================

TAILLE_CIBLE = (32, 32)

def preprocess(x_hwc4: np.ndarray) -> np.ndarray:

    x = np.asarray(x_hwc4, dtype=np.float32)

    if x.ndim != 3 or x.shape[-1] != 4:
        raise ValueError(f"Attendu (H,W,4), reçu {x.shape}")

    # 🔥 AJOUT IMPORTANT
    x = np.nan_to_num(x, nan=0.0)

    # Normalisation par canal
    for c in range(x.shape[2]):
        max_abs = np.max(np.abs(x[:, :, c]))
        if max_abs > 0:
            x[:, :, c] /= max_abs

    resized = np.zeros((*TAILLE_CIBLE, x.shape[2]), dtype=np.float32)

    for c in range(x.shape[2]):
        pil_img = Image.fromarray(x[:, :, c], mode="F")
        pil_r = pil_img.resize(
            (TAILLE_CIBLE[1], TAILLE_CIBLE[0]),
            resample=Image.BILINEAR
        )
        resized[:, :, c] = np.array(pil_r, dtype=np.float32)

    flat = resized.flatten()

    # 🔥 Sécurité ultime
    flat = np.nan_to_num(flat, nan=0.0)

    return flat


# ============================================================
# PREDICTIONS
# ============================================================

def predict_task1(x: np.ndarray, models: dict, threshold: float) -> dict:
    pipeline = models["task1_pipeline"]
    x = x.reshape(1, -1)
    proba = pipeline.predict_proba(x)[0][1]
    pred = int(proba >= threshold)
    return {"class": pred, "probability": float(proba)}


# Mock others
def _mock_binary(seed: str, threshold: float) -> dict:
    rng = np.random.default_rng(abs(hash(seed)) % (2**32))
    proba = float(rng.uniform(0, 1))
    return {"class": int(proba >= threshold), "probability": proba}


def _mock_reg(seed: str) -> dict:
    rng = np.random.default_rng(abs(hash(seed)) % (2**32))
    return {"width_m": float(rng.uniform(5, 80))}


# ============================================================
# MAIN ANALYSIS
# ============================================================

def run_analysis(files, cfg, models):
    results = []
    progress = st.progress(0)

    for i, f in enumerate(files):
        name = f.name
        row = {"filename": name}

        try:
            x = load_npz_to_hwc4(f)
            x_p = preprocess(x)

            t1 = predict_task1(x_p, models, cfg.thresholds.t1_presence)
            t2 = _mock_reg(name)
            t3 = _mock_binary(name + "_t3", cfg.thresholds.t3_current)
            t4 = _mock_binary(name + "_t4", cfg.thresholds.t4_parallel)

            row.update({
                "t1_pred": t1["class"],
                "t1_proba": t1["probability"],
                "t2_width_m": t2["width_m"],
                "t3_pred": t3["class"],
                "t3_proba": t3["probability"],
                "t4_pred": t4["class"],
                "t4_proba": t4["probability"],
                "ok": True
            })

        except Exception as e:
            row.update({"ok": False, "error": str(e)})

        results.append(row)
        progress.progress((i + 1) / len(files))

    progress.empty()
    return results


# ============================================================
# UI
# ============================================================

def main():
    st.set_page_config(layout="wide")
    st.title(APP_NAME)
    st.caption(TAGLINE)

    with st.sidebar:
        f = st.file_uploader("Upload .npz", type=["npz"], accept_multiple_files=True)
        t1_thr = st.slider("Seuil T1", 0.0, 1.0, 0.4, 0.01)
        run = st.button("Analyse", type="primary")

    if not f:
        st.info("Upload un fichier .npz")
        return

    models = load_models()
    cfg = RunConfig(thresholds=Thresholds(t1_presence=t1_thr))

    if run:
        results = run_analysis(f, cfg, models)
        st.session_state["results"] = results

    if "results" not in st.session_state:
        return

    results = st.session_state["results"]

    st.subheader("Résultats")
    df = pd.DataFrame(results) if pd else results
    st.dataframe(df)

    if pd:
        st.download_button(
            "Télécharger CSV",
            df.to_csv(index=False).encode(),
            "results.csv"
        )


if __name__ == "__main__":
    main()
