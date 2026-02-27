from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import streamlit as st

try:
    import pandas as pd  # optional but handy
except Exception:
    pd = None  # type: ignore

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # type: ignore


APP_NAME = "Magnetic Map — Analysis Console"
TAGLINE = "Managing the invisible — Analysis Console"
ACCENT_HEX = "#FF002F"


@dataclass(frozen=True)
class Thresholds:
    t1_pipeline_presence: float = 0.5
    t3_current_sufficient: float = 0.5
    t4_parallel_pipelines: float = 0.5


@dataclass(frozen=True)
class RunConfig:
    thresholds: Thresholds
    device: str = "cpu"      # placeholder: "cpu" / "cuda"
    mock_mode: bool = True   # UI test mode when models are not wired yet


def _normalize_key(k: str) -> str:
    # normalize "Bx", "b_x", "B-X" -> "bx"
    return "".join(ch for ch in k.lower().strip() if ch.isalnum())


def _stable_rng(seed_source: str) -> np.random.Generator:
    # deterministic RNG so UI is stable during demos
    h = hashlib.md5(seed_source.encode("utf-8")).hexdigest()[:16]
    seed = int(h, 16) % (2**32)
    return np.random.default_rng(seed)


def _robust_display_scale(a: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Scale to [0,1] for display based on robust percentiles."""
    a = np.asarray(a, dtype=np.float32)
    if not np.isfinite(a).any():
        return np.zeros_like(a, dtype=np.float32)

    lo = float(np.nanpercentile(a, p_low))
    hi = float(np.nanpercentile(a, p_high))
    if hi <= lo:
        return np.zeros_like(a, dtype=np.float32)

    out = (a - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def load_npz_to_hwc4(uploaded_file: Any) -> np.ndarray:
    """
    Load a .npz (UploadedFile or file-like) and return float32 array (H, W, 4).
    Supports:
      - Single array shaped (H, W, 4) or (4, H, W)
      - Four arrays named (Bx, By, Bz, Norm) (case/underscore-insensitive)
    """
    # Streamlit UploadedFile is file-like; ensure pointer at start.
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    with np.load(uploaded_file, allow_pickle=False) as z:
        keys = list(getattr(z, "files", []))

        # Case 1: explicit channels
        want = {"bx": None, "by": None, "bz": None, "norm": None}
        norm_map = {_normalize_key(k): k for k in keys}

        for short in list(want.keys()):
            if short in norm_map:
                want[short] = norm_map[short]

        if all(v is not None for v in want.values()):
            arrays = [np.asarray(z[want[k]]) for k in ("bx", "by", "bz", "norm")]
            arrays2d: List[np.ndarray] = []
            for a in arrays:
                a = np.asarray(a)
                if a.ndim == 3 and 1 in a.shape:
                    a = np.squeeze(a)
                if a.ndim != 2:
                    raise ValueError(f"Canal attendu 2D, reçu shape={a.shape}")
                arrays2d.append(a.astype(np.float32, copy=False))

            shapes = {a.shape for a in arrays2d}
            if len(shapes) != 1:
                raise ValueError(f"Shapes canaux incohérentes: {sorted(shapes)}")

            return np.stack(arrays2d, axis=-1)  # (H,W,4)

        # Case 2: single array (arr_0, image, etc.)
        if len(keys) == 0:
            raise ValueError("Archive .npz vide (aucun array).")

        preferred = ["image", "arr_0", "data", "x"]
        candidate_keys = [k for k in preferred if k in keys] + [k for k in keys if k not in preferred]

        for k in candidate_keys:
            a = np.asarray(z[k])
            if a.ndim == 3 and a.shape[-1] == 4:
                return a.astype(np.float32, copy=False)
            if a.ndim == 3 and a.shape[0] == 4:
                return np.moveaxis(a, 0, -1).astype(np.float32, copy=False)

        raise ValueError("Contenu .npz non reconnu. Attendu (H,W,4) ou Bx/By/Bz/Norm.")


@st.cache_resource(show_spinner=False)
def load_models() -> Dict[str, Any]:
    """
    Placeholder model loader.
    Replace this with your real code (PyTorch, ONNX, etc.).
    Tip: keep a dict {task1: model1, task2: model2, ...}.
    """
    return {"task1": None, "task2": None, "task3": None, "task4": None}


def preprocess(x_hwc4: np.ndarray) -> np.ndarray:
    """
    Placeholder preprocessing:
    - enforce float32
    - insert your resizing/normalization here if your models require fixed shapes.
    """
    x = np.asarray(x_hwc4, dtype=np.float32)
    if x.ndim != 3 or x.shape[-1] != 4:
        raise ValueError(f"Attendu (H,W,4), reçu {x.shape}")
    return x


def _mock_binary(head: str, threshold: float, seed: str) -> Dict[str, Any]:
    rng = _stable_rng(seed + ":" + head)
    proba = float(rng.uniform(0.0, 1.0))
    pred = int(proba >= threshold)
    return {"pred": pred, "proba": proba, "threshold": threshold}


def _mock_regression(seed: str) -> Dict[str, Any]:
    rng = _stable_rng(seed + ":reg")
    width = float(rng.uniform(5.0, 80.0))
    return {"width_m": width}


def predict_task1(x: np.ndarray, model: Any, *, threshold: float, cfg: RunConfig, seed: str) -> Dict[str, Any]:
    if cfg.mock_mode or model is None:
        return _mock_binary("t1", threshold, seed)
    raise NotImplementedError("Brancher ici l’inférence réelle pour Tâche 1.")


def predict_task2(x: np.ndarray, model: Any, *, cfg: RunConfig, seed: str) -> Dict[str, Any]:
    if cfg.mock_mode or model is None:
        return _mock_regression(seed)
    raise NotImplementedError("Brancher ici l’inférence réelle pour Tâche 2.")


def predict_task3(x: np.ndarray, model: Any, *, threshold: float, cfg: RunConfig, seed: str) -> Dict[str, Any]:
    if cfg.mock_mode or model is None:
        return _mock_binary("t3", threshold, seed)
    raise NotImplementedError("Brancher ici l’inférence réelle pour Tâche 3.")


def predict_task4(x: np.ndarray, model: Any, *, threshold: float, cfg: RunConfig, seed: str) -> Dict[str, Any]:
    if cfg.mock_mode or model is None:
        return _mock_binary("t4", threshold, seed)
    raise NotImplementedError("Brancher ici l’inférence réelle pour Tâche 4.")


def _label_binary(task: str, pred: int) -> str:
    if task == "t1":
        return "Présence" if pred == 1 else "Absence"
    if task == "t3":
        return "Suffisante" if pred == 1 else "Insuffisante"
    if task == "t4":
        return "Parallèles" if pred == 1 else "Unique/Absence"
    return str(pred)


def _ensure_state() -> None:
    st.session_state.setdefault("results", None)
    st.session_state.setdefault("last_run_ts", None)
    st.session_state.setdefault("input_fp", None)


def _reset_results() -> None:
    st.session_state["results"] = None
    st.session_state["last_run_ts"] = None


def _plot_preview(x: np.ndarray) -> None:
    if plt is None:
        st.warning("Matplotlib indisponible: preview désactivée.")
        return

    titles = ["Bx", "By", "Bz", "Norm"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for i, ax in enumerate(axes.flat):
        img = _robust_display_scale(x[:, :, i])
        ax.imshow(img)
        ax.set_title(titles[i])
        ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


def _results_to_dataframe(rows: List[Dict[str, Any]]):
    if pd is None:
        return rows
    return pd.DataFrame(rows)


def _export_csv(df_or_rows) -> bytes:
    if pd is not None and hasattr(df_or_rows, "to_csv"):
        return df_or_rows.to_csv(index=False).encode("utf-8")

    # fallback without pandas
    import csv
    import io as _io
    rows = df_or_rows
    if not rows:
        return b""
    buf = _io.StringIO()
    fieldnames = sorted({k for r in rows for k in r.keys()})
    w = csv.DictWriter(buf, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode("utf-8")


def _export_json(rows: List[Dict[str, Any]]) -> bytes:
    return json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8")


def run_analysis(files: List[Any], cfg: RunConfig, models: Dict[str, Any]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    total = max(len(files), 1)

    progress = st.progress(0, text="Analyse en cours…")
    status_slot = st.empty()

    for i, f in enumerate(files, start=1):
        name = getattr(f, "name", f"file_{i}.npz")
        status_slot.write(f"• {name}")

        row: Dict[str, Any] = {"filename": name, "ok": False, "error": None}

        try:
            x = load_npz_to_hwc4(f)
            x_in = preprocess(x)

            seed = name  # stable per file; customize if needed
            t1 = predict_task1(x_in, models.get("task1"), threshold=cfg.thresholds.t1_pipeline_presence, cfg=cfg, seed=seed)
            t2 = predict_task2(x_in, models.get("task2"), cfg=cfg, seed=seed)
            t3 = predict_task3(x_in, models.get("task3"), threshold=cfg.thresholds.t3_current_sufficient, cfg=cfg, seed=seed)
            t4 = predict_task4(x_in, models.get("task4"), threshold=cfg.thresholds.t4_parallel_pipelines, cfg=cfg, seed=seed)

            row.update({
                "t1_pred": int(t1["pred"]),
                "t1_proba": float(t1["proba"]),
                "t2_width_m": float(t2["width_m"]),
                "t3_pred": int(t3["pred"]),
                "t3_proba": float(t3["proba"]),
                "t4_pred": int(t4["pred"]),
                "t4_proba": float(t4["proba"]),
                "ok": True,
            })

        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"

        results.append(row)
        progress.progress(i / total, text=f"Analyse: {i}/{total}")

    time.sleep(0.1)
    progress.empty()
    status_slot.empty()
    return results


def main() -> None:
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="🧲",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _ensure_state()

    st.markdown(
        f"""
        <style>
          .app-header {{
            padding: 0.6rem 0 0.2rem 0;
          }}
          .tagline {{
            opacity: 0.82;
            font-size: 0.95rem;
            margin-top: -0.25rem;
          }}
          /* Console-like metric cards */
          div[data-testid="metric-container"] {{
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 0.8rem 0.8rem;
            border-radius: 12px;
          }}
        </style>
        <div class="app-header">
          <h1 style="margin-bottom: 0.1rem;">{APP_NAME}</h1>
          <div class="tagline">{TAGLINE}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Entrée")
        mode = st.radio("Mode d’upload", ["Fichier .npz", "Dossier .npz (batch)"], index=0)

        uploaded_files: List[Any] = []
        if mode == "Fichier .npz":
            f = st.file_uploader("Choisis un fichier .npz", type=["npz"], accept_multiple_files=False)
            if f is not None:
                uploaded_files = [f]
        else:
            # Try directory mode (newer Streamlit), fallback to multi-file.
            try:
                fs = st.file_uploader(
                    "Choisis un dossier (ou plusieurs fichiers)",
                    type=["npz"],
                    accept_multiple_files="directory",
                )
            except TypeError:
                fs = st.file_uploader(
                    "Upload dossier non supporté ici : sélectionne plusieurs fichiers .npz",
                    type=["npz"],
                    accept_multiple_files=True,
                )
            if fs:
                uploaded_files = list(fs)

        # If inputs changed, drop stale results (avoid UX confusion).
        names = [getattr(f, "name", "") for f in uploaded_files]
        fp = "|".join(names)
        if st.session_state.get("input_fp") != fp:
            _reset_results()
            st.session_state["input_fp"] = fp

        st.markdown("### Paramètres")
        mock = st.toggle("Mode démo (mock)", value=True, help="Désactive quand tes modèles réels sont branchés.")
        device = st.selectbox("Device", options=["cpu", "cuda"], index=0)

        st.markdown("#### Seuils de décision")
        t1_thr = st.slider("Tâche 1 — Présence de conduite", 0.0, 1.0, 0.5, 0.01)
        t3_thr = st.slider("Tâche 3 — Intensité suffisante", 0.0, 1.0, 0.5, 0.01)
        t4_thr = st.slider("Tâche 4 — Conduites parallèles", 0.0, 1.0, 0.5, 0.01)

        cfg = RunConfig(
            thresholds=Thresholds(
                t1_pipeline_presence=t1_thr,
                t3_current_sufficient=t3_thr,
                t4_parallel_pipelines=t4_thr,
            ),
            device=device,
            mock_mode=mock,
        )

        st.markdown("### Action")
        st.button("Reset", on_click=_reset_results, type="secondary", use_container_width=True)

        run = st.button(
            "Analyse",
            type="primary",
            use_container_width=True,
            disabled=(len(uploaded_files) == 0),
        )
        st.caption("Commence ‘single file’, puis passe en batch. Next: perf + reporting. 🚀")

    if len(uploaded_files) == 0:
        st.info("Upload un fichier ou un dossier .npz pour démarrer.")
        return

    # Preview selection
    names = [getattr(f, "name", f"file_{i}.npz") for i, f in enumerate(uploaded_files, start=1)]
    selection = st.selectbox("Preview", options=names, index=0)
    sel_file = uploaded_files[names.index(selection)]

    with st.expander("Preview des 4 canaux", expanded=True):
        try:
            x_preview = load_npz_to_hwc4(sel_file)
            st.caption(f"Shape: {x_preview.shape} (H, W, 4)")
            _plot_preview(x_preview)
        except Exception as e:
            st.error(f"Impossible de prévisualiser ce fichier: {type(e).__name__}: {e}")

    models = load_models()

    if run:
        rows = run_analysis(uploaded_files, cfg, models)
        st.session_state["results"] = rows
        st.session_state["last_run_ts"] = time.time()

    rows = st.session_state.get("results")
    if not rows:
        st.warning("Résultats vides. Clique sur Analyse.")
        return

    st.subheader("Résultats")
    total = len(rows)
    ok = sum(1 for r in rows if r.get("ok"))
    err = total - ok

    c1, c2, c3 = st.columns(3)
    c1.metric("Fichiers", total)
    c2.metric("OK", ok)
    c3.metric("Erreurs", err)

    # Single-file summary
    focus = next((r for r in rows if r.get("filename") == selection), None)
    if focus and focus.get("ok"):
        st.markdown("#### Résumé du fichier sélectionné")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tâche 1", _label_binary("t1", int(focus["t1_pred"])), f"p={focus['t1_proba']:.2f}")
        m2.metric("Tâche 2", f"{focus['t2_width_m']:.2f} m", "largeur estimée")
        m3.metric("Tâche 3", _label_binary("t3", int(focus["t3_pred"])), f"p={focus['t3_proba']:.2f}")
        m4.metric("Tâche 4", _label_binary("t4", int(focus["t4_pred"])), f"p={focus['t4_proba']:.2f}")
    elif focus:
        st.markdown("#### Résumé du fichier sélectionné")
        st.error(focus.get("error") or "Erreur inconnue.")

    df_or_rows = _results_to_dataframe(rows)
    st.markdown("#### Tableau batch")
    st.dataframe(df_or_rows, use_container_width=True)

    st.markdown("#### Export")
    csv_bytes = _export_csv(df_or_rows)
    json_bytes = _export_json(rows)

    d1, d2 = st.columns(2)
    d1.download_button(
        "Télécharger CSV",
        data=csv_bytes,
        file_name="results.csv",
        mime="text/csv",
        use_container_width=True,
    )
    d2.download_button(
        "Télécharger JSON",
        data=json_bytes,
        file_name="results.json",
        mime="application/json",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
