from __future__ import annotations

import time
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

APP_NAME = "Magnetic Map — Analysis Console"
TAGLINE = "Managing the invisible"


# ============================================================
# CONFIG
# ============================================================

TAILLE_CIBLE = (32, 32)
SEUIL_DEFAUT = 0.4


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_models() -> Dict[str, Any]:
    return {
        "task1_pipeline": joblib.load(Path("models/model_task1.pkl"))
    }


# ============================================================
# LOAD NPZ
# ============================================================

def load_npz_to_hwc4(uploaded_file: Any) -> np.ndarray:
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    with np.load(uploaded_file, allow_pickle=False) as z:
        keys = list(z.files)

        # Cas Bx,By,Bz,Norm
        channels = {}
        for k in keys:
            k_norm = "".join(c for c in k.lower() if c.isalnum())
            if k_norm in ["bx", "by", "bz", "norm"]:
                channels[k_norm] = z[k]

        if len(channels) == 4:
            arrays = [np.asarray(channels[k]) for k in ["bx", "by", "bz", "norm"]]
            arrays = [np.squeeze(a).astype(np.float32) for a in arrays]
            return np.stack(arrays, axis=-1)

        # Cas array unique
        for k in keys:
            a = np.asarray(z[k])
            if a.ndim == 3 and a.shape[-1] == 4:
                return a.astype(np.float32)
            if a.ndim == 3 and a.shape[0] == 4:
                return np.moveaxis(a, 0, -1).astype(np.float32)

        raise ValueError("Format .npz non reconnu.")


# ============================================================
# PREPROCESS (IDENTIQUE TRAINING)
# ============================================================

def preprocess(x_hwc4: np.ndarray) -> np.ndarray:
    x = np.asarray(x_hwc4, dtype=np.float32)

    if x.ndim != 3 or x.shape[-1] != 4:
        raise ValueError(f"Attendu (H,W,4), reçu {x.shape}")

    # Remplacement NaN
    x = np.nan_to_num(x, nan=0.0)

    # Normalisation par canal
    for c in range(x.shape[2]):
        max_abs = np.max(np.abs(x[:, :, c]))
        if max_abs > 0:
            x[:, :, c] /= max_abs

    resized = np.zeros((*TAILLE_CIBLE, 4), dtype=np.float32)

    for c in range(4):
        pil_img = Image.fromarray(x[:, :, c], mode="F")
        pil_r = pil_img.resize(
            (TAILLE_CIBLE[1], TAILLE_CIBLE[0]),
            resample=Image.BILINEAR
        )
        resized[:, :, c] = np.array(pil_r, dtype=np.float32)

    flat = resized.flatten()
    flat = np.nan_to_num(flat, nan=0.0)

    return flat


# ============================================================
# PREDICTION TASK 1
# ============================================================

def predict_task1(x: np.ndarray, models: dict, threshold: float) -> dict:
    pipeline = models["task1_pipeline"]

    x = x.reshape(1, -1)
    proba = pipeline.predict_proba(x)[0][1]
    pred = int(proba >= threshold)

    return {
        "class": pred,
        "probability": float(proba)
    }


# ============================================================
# ANALYSIS
# ============================================================

def run_analysis(files: List[Any], threshold: float, models: Dict[str, Any]):
    results = []
    progress = st.progress(0)

    for i, f in enumerate(files):
        name = f.name
        row = {"filename": name}

        try:
            x = load_npz_to_hwc4(f)
            x_p = preprocess(x)

            t1 = predict_task1(x_p, models, threshold)

            row.update({
                "Présence détectée": "OUI" if t1["class"] == 1 else "NON",
                "Confiance (%)": round(t1["probability"] * 100, 1),
                "ok": True
            })

        except Exception as e:
            row.update({
                "Présence détectée": "-",
                "Confiance (%)": "-",
                "ok": False,
                "error": str(e)
            })

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
        uploaded = st.file_uploader(
            "Upload un ou plusieurs fichiers .npz",
            type=["npz"],
            accept_multiple_files=True
        )

        threshold = st.slider(
            "Seuil de décision",
            0.0, 1.0,
            SEUIL_DEFAUT,
            0.01
        )

        run = st.button("Analyse", type="primary")

    if not uploaded:
        st.info("Upload un fichier pour démarrer.")
        return

    models = load_models()

    if run:
        results = run_analysis(uploaded, threshold, models)
        st.session_state["results"] = results

    if "results" not in st.session_state:
        return

    results = st.session_state["results"]

    st.subheader("Résultats")

    if pd:
        df = pd.DataFrame(results)
        st.dataframe(
            df[["filename", "Présence détectée", "Confiance (%)"]],
            use_container_width=True
        )

        st.download_button(
            "Télécharger CSV",
            df.to_csv(index=False).encode(),
            "results.csv"
        )

    # Résumé du premier fichier
    focus = results[0]
    if focus["ok"]:
        st.markdown("### Résumé")

        col1, col2 = st.columns(2)

        col1.metric("Présence de conduite", focus["Présence détectée"])
        col2.metric("Confiance modèle", f"{focus['Confiance (%)']} %")

        if 40 < focus["Confiance (%)"] < 60:
            st.warning("Prédiction incertaine (zone grise).")

    else:
        st.error(focus.get("error", "Erreur inconnue"))


if __name__ == "__main__":
    main()
