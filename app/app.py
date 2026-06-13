from __future__ import annotations

import sys
import os
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(os.getenv("SMART_PRODUCT_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_csv, load_parquet
from src.models import artifact_status


DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = Path(os.getenv("SMART_PRODUCT_PROCESSED_DIR", DATA_DIR / "processed"))
REPORTS_DIR = Path(os.getenv("SMART_PRODUCT_REPORTS_DIR", PROJECT_ROOT / "reports"))
MODELS_DIR = Path(os.getenv("SMART_PRODUCT_MODELS_DIR", PROJECT_ROOT / "models"))


st.set_page_config(page_title="Smart Product Intelligence", layout="wide")
st.title("Smart Product Intelligence")


@st.cache_data(show_spinner=False)
def safe_csv(path: Path) -> pd.DataFrame | None:
    try:
        return load_csv(path)
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def safe_parquet(path: Path) -> pd.DataFrame | None:
    try:
        return load_parquet(path)
    except FileNotFoundError:
        return None


product_test = safe_parquet(PROCESSED_DIR / "product_test.parquet")
review_test = safe_parquet(PROCESSED_DIR / "test.parquet")
tabular_metrics = safe_csv(REPORTS_DIR / "milestone1_tabular_mlp_metrics.csv")
vision_metrics = safe_csv(REPORTS_DIR / "milestone2_vision_metrics.csv")
transformer_metrics = safe_csv(REPORTS_DIR / "milestone4_transformer_metrics.csv")
rag_metrics = safe_csv(REPORTS_DIR / "milestone5_rag_grounding_metrics.csv")
diffusion_manifest = safe_csv(REPORTS_DIR / "milestone6_generated_image_manifest.csv")

status = artifact_status(MODELS_DIR)
cols = st.columns(4)
for col, (name, exists) in zip(cols, status.items()):
    col.metric(name.replace("_", " ").title(), "Ready" if exists else "Missing")

st.divider()

left, right = st.columns([1, 1])
with left:
    st.subheader("Data Snapshot")
    if product_test is not None:
        st.write(f"Test products: {len(product_test):,}")
        st.dataframe(
            product_test[["product_id", "product_title", "rating_band", "target_rating"]].head(20),
            use_container_width=True,
        )
    else:
        st.info("Run `notebooks/00_eda.ipynb` to create processed product splits.")

with right:
    st.subheader("Review Snapshot")
    if review_test is not None:
        st.write(f"Test reviews: {len(review_test):,}")
        st.dataframe(
            review_test[["product_id", "title", "sentiment_label", "review_text"]].head(20),
            use_container_width=True,
        )
    else:
        st.info("Run `notebooks/00_eda.ipynb` to create processed review splits.")

st.divider()
st.subheader("Milestone Metrics")
tabs = st.tabs(["Tabular", "Vision", "Transformer", "RAG", "Diffusion"])

for tab, label, frame in [
    (tabs[0], "Milestone 1", tabular_metrics),
    (tabs[1], "Milestone 2", vision_metrics),
    (tabs[2], "Milestone 4", transformer_metrics),
    (tabs[3], "Milestone 5", rag_metrics),
    (tabs[4], "Milestone 6", diffusion_manifest),
]:
    with tab:
        if frame is not None:
            st.dataframe(frame, use_container_width=True)
        else:
            st.info(f"Run the relevant notebook to create {label} outputs.")
