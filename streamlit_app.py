from __future__ import annotations

import time
from typing import Any, Dict, List
import requests
import numpy as np
import streamlit as st
import joblib
import tensorflow as tf
import gdown
from scipy.ndimage import zoom
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

TARGET_SIZE = (224,224)
SEUIL_DEFAUT = 0.5
MODEL_ID = "1nt44M2ut14aU1wXm2WgBqfTjGZCrkEPt"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_models():

    local_path = "models/modele_tache1.keras"
    Path("models").mkdir(exist_ok=True)

    if not Path(local_path).exists():
        url = f"https://drive.google.com/uc?id={MODEL_ID}"
        gdown.download(url, local_path, quiet=False)

    model = tf.keras.models.load_model(local_path)

    return {"task1_model": model}
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
    """
    Reproduction EXACTE du preprocessing training.
    """

    # float32 + NaN -> 0
    img = np.nan_to_num(x_hwc4.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # Normalisation par canal
    max_abs = np.abs(img).max(axis=(0, 1))
    np.maximum(max_abs, 1e-12, out=max_abs)
    img = img / max_abs

    # Resize identique training
    h, w = img.shape[:2]
    factors = (TARGET_SIZE[0] / h, TARGET_SIZE[1] / w, 1)

    img_resized = zoom(img, factors, order=1)

      # 👇 DEBUG TEMPORAIRE
    st.write("Min:", img_resized.min())
    st.write("Max:", img_resized.max())
    st.write("Mean:", img_resized.mean())
    st.write("Shape:", img_resized.shape)

    return img_resized.astype(np.float32)


# ============================================================
# PREDICTION TASK 1
# ============================================================

def predict_task1(x: np.ndarray, models: dict, threshold: float):

    model = models["task1_model"]

    x = np.expand_dims(x, axis=0)  # (1,224,224,4)

    proba = model.predict(x, verbose=0)[0][0]
    
    #DEBUG
    st.write("Proba brute:", proba)

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
