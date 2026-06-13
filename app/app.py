from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(os.getenv("SMART_PRODUCT_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = Path(os.getenv("SMART_PRODUCT_PROCESSED_DIR", DATA_DIR / "processed"))
REPORTS_DIR = Path(os.getenv("SMART_PRODUCT_REPORTS_DIR", PROJECT_ROOT / "reports"))
MODELS_DIR = Path(os.getenv("SMART_PRODUCT_MODELS_DIR", PROJECT_ROOT / "models"))

PRODUCT_LEVEL_CANDIDATES = [
    PROCESSED_DIR / "product_level_with_split.parquet",
    PROCESSED_DIR / "product_level.parquet",
    PROCESSED_DIR / "product_test.parquet",
]
REVIEW_LEVEL_CANDIDATES = [
    PROCESSED_DIR / "review_level.parquet",
    PROCESSED_DIR / "test.parquet",
]

TABULAR_MODEL_PATH = MODELS_DIR / "tabular_mlp.keras"
TABULAR_PREPROCESSOR_CANDIDATES = [
    MODELS_DIR / "tabular_mlp_preprocessor.joblib",
    MODELS_DIR / "tabular_mlp_pipeline.joblib",
]
TABULAR_PIPELINE_PATH = MODELS_DIR / "tabular_mlp_pipeline.joblib"
TABULAR_LABEL_ENCODER_PATH = MODELS_DIR / "tabular_mlp_label_encoder.joblib"
VISION_MODEL_CANDIDATES = [
    MODELS_DIR / "small_cnn.keras",
    MODELS_DIR / "vision_transfer.keras",
    MODELS_DIR / "vision_mobilenet.keras",
    MODELS_DIR / "vision_mobilenetv2_transfer.keras",
    MODELS_DIR / "vision_transfer_head.keras",
    MODELS_DIR / "vision_small_cnn.keras",
]
VISION_LABEL_ENCODER_PATH = MODELS_DIR / "vision_label_encoder.joblib"
EMBEDDING_CANDIDATES = [
    PROCESSED_DIR / "product_text_embeddings.npy",
    PROCESSED_DIR / "product_text_search_index.csv",
    MODELS_DIR / "product_text_embeddings.npy",
    MODELS_DIR / "product_text_index.faiss",
]
GENERATED_IMAGE_DIRS = [
    DATA_DIR / "generated_images",
    PROJECT_ROOT / "outputs" / "generated_images",
    REPORTS_DIR / "generated_images",
    REPORTS_DIR / "milestone6_generated_images",
]


st.set_page_config(page_title="Smart Product Intelligence Demo", layout="wide")


def first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def fallback_products() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product_id": "ARTIFACTS_MISSING",
                "product_title": "Sample product placeholder - load Milestone 0 data for real products",
                "store_name": "Not available",
                "price_num": np.nan,
                "average_rating": np.nan,
                "rating_number": np.nan,
                "review_count": np.nan,
                "description": "Product-level parquet was not found. The demo shell is still shown so Milestone 7 can run without crashing.",
                "_is_placeholder": True,
            }
        ]
    )


def display_value(value, fallback: str = "Not available") -> str:
    if pd.isna(value) or value == "":
        return fallback
    return str(value)


def resolve_path(path_value) -> Path | None:
    if pd.isna(path_value) or not str(path_value).strip():
        return None
    path = Path(str(path_value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def product_text(row: pd.Series) -> str:
    parts = []
    for col in ["product_title", "description", "features", "details", "store_name", "main_category_clean"]:
        if col in row.index and not pd.isna(row[col]):
            parts.append(str(row[col]))
    return " ".join(parts)


def clean_snippet(text: str, max_len: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


@st.cache_data(show_spinner=False)
def load_parquet_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_csv_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_joblib_safe(path: Path):
    if not path.exists():
        return None
    import joblib

    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_keras_model_safe(path: Path):
    if not path.exists():
        return None
    try:
        import tensorflow as tf

        return tf.keras.models.load_model(path)
    except Exception as exc:
        return exc


@st.cache_data(show_spinner=False)
def build_similarity_index(products: pd.DataFrame, max_rows: int = 5000):
    base = products.copy()
    if len(base) > max_rows:
        base = base.sample(max_rows, random_state=42)
    base["similarity_text"] = base.apply(product_text, axis=1).fillna("")
    vectorizer = TfidfVectorizer(max_features=8000, stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(base["similarity_text"])
    return base.reset_index(drop=True), vectorizer, matrix


def find_report(*relative_paths: str) -> pd.DataFrame | None:
    for rel in relative_paths:
        frame = load_csv_safe(PROJECT_ROOT / rel)
        if frame is not None:
            return frame
    return None


def image_source(row: pd.Series):
    cached_path = resolve_path(row.get("cached_image_path"))
    if cached_path and cached_path.exists():
        return cached_path
    image_url = row.get("primary_image_url")
    if not pd.isna(image_url) and str(image_url).startswith(("http://", "https://")):
        return str(image_url)
    return None


def show_product_image(row: pd.Series, caption: str | None = None) -> None:
    source = image_source(row)
    if source:
        st.image(source, caption=caption, width="stretch")
    else:
        st.warning("Product image artifact not found. Run Milestone 0 image caching first.")


def show_dataframe_or_warning(frame: pd.DataFrame | None, message: str) -> None:
    if frame is None or frame.empty:
        st.warning(message)
    else:
        st.dataframe(frame, width="stretch")


def select_product(products: pd.DataFrame) -> pd.Series | None:
    st.subheader("Product Selector")
    if products.empty:
        st.warning("Product data is empty. Run Milestone 0 first.")
        return None

    selector_df = products.dropna(subset=["product_id"]).copy()
    if len(selector_df) > 1000:
        selector_df = selector_df.sample(1000, random_state=42)
    selector_df = selector_df.sort_values("product_title", na_position="last").reset_index(drop=True)
    selector_df["label"] = selector_df.apply(
        lambda row: f"{display_value(row.get('product_title'), 'Untitled product')} [{row.get('product_id')}]",
        axis=1,
    )

    typed_id = st.text_input("Or enter a product ID", placeholder="Example: B00JQRPB9I")
    if typed_id:
        exact = products[products["product_id"].astype(str).str.lower() == typed_id.strip().lower()]
        if not exact.empty:
            return exact.iloc[0]
        st.warning("Product ID not found in product-level data.")

    label = st.selectbox("Choose a product", selector_df["label"].tolist())
    selected_id = selector_df.loc[selector_df["label"] == label, "product_id"].iloc[0]
    return products[products["product_id"].astype(str) == str(selected_id)].iloc[0]


def product_information(row: pd.Series) -> None:
    st.subheader("Product Information")
    left, right = st.columns([1, 2])
    with left:
        show_product_image(row, caption=display_value(row.get("product_title"), "Selected product"))
    with right:
        st.markdown(f"### {display_value(row.get('product_title'), 'Untitled product')}")
        metrics = st.columns(4)
        metrics[0].metric("Product ID", display_value(row.get("product_id")))
        metrics[1].metric("Store", display_value(row.get("store_name")))
        price = row.get("price_num")
        metrics[2].metric("Price", "Not available" if pd.isna(price) else f"${float(price):.2f}")
        metrics[3].metric("Avg Rating", display_value(row.get("average_rating")))
        st.write(f"Rating number: {display_value(row.get('rating_number', row.get('review_count')))}")
        st.write(f"Review count: {display_value(row.get('review_count'))}")
        description = product_text(row)
        if description:
            st.caption(clean_snippet(description, 900))
        else:
            st.warning("Product description/features artifact not found in product-level data.")


def tabular_section(row: pd.Series) -> None:
    st.subheader("Milestone 1 - Tabular Rating Prediction")
    metrics = find_report(
        "reports/milestone1_tabular_mlp_metrics.csv",
        "reports/tabular_mlp_model_comparison.csv",
        "report/tabular_mlp_model_comparison.csv",
    )
    show_dataframe_or_warning(metrics, "Tabular comparison CSV not found. Run Milestone 1 first.")

    if not TABULAR_MODEL_PATH.exists():
        st.warning("Artifact not found: models/tabular_mlp.keras. Run Milestone 1 first.")
        return

    pipeline = load_joblib_safe(TABULAR_PIPELINE_PATH)
    label_encoder = load_joblib_safe(TABULAR_LABEL_ENCODER_PATH)
    if pipeline is None or label_encoder is None:
        preprocessor_path = first_existing(TABULAR_PREPROCESSOR_CANDIDATES)
        if preprocessor_path is None:
            st.warning("Live prediction requires a tabular preprocessing artifact, for example models/tabular_mlp_preprocessor.joblib.")
        st.warning("Live prediction requires models/tabular_mlp_pipeline.joblib and models/tabular_mlp_label_encoder.joblib.")
        return

    try:
        pred = pipeline.predict(pd.DataFrame([row.to_dict()]))
        proba = pipeline.predict_proba(pd.DataFrame([row.to_dict()])) if hasattr(pipeline, "predict_proba") else None
        label = pred[0]
        if hasattr(label_encoder, "inverse_transform") and not isinstance(label, str):
            label = label_encoder.inverse_transform([label])[0]
        st.success(f"Predicted rating band: {label}")
        if proba is not None:
            st.write(f"Confidence: {float(np.max(proba)):.3f}")
    except Exception as exc:
        st.warning(f"Live tabular prediction could not run with saved artifacts: {exc}")


def vision_section(row: pd.Series) -> None:
    st.subheader("Milestone 2 - Vision Prediction")
    show_product_image(row)
    metrics = find_report(
        "reports/milestone2_vision_metrics.csv",
        "reports/vision_model_comparison.csv",
        "report/vision_model_comparison.csv",
    )
    show_dataframe_or_warning(metrics, "Vision comparison CSV not found. Run Milestone 2 first.")

    model_path = first_existing(VISION_MODEL_CANDIDATES)
    if model_path is None:
        st.warning("Vision model artifact not found. Run Milestone 2 first.")
        return

    source = image_source(row)
    if not source or isinstance(source, str):
        st.warning("Live vision prediction requires a locally cached product image.")
        return

    model = load_keras_model_safe(model_path)
    if isinstance(model, Exception):
        st.warning(f"Vision model could not be loaded: {model}")
        return

    try:
        from PIL import Image

        image = Image.open(source).convert("RGB").resize((224, 224))
        array = np.asarray(image, dtype=np.float32)[None, ...] / 255.0
        pred = model.predict(array, verbose=0)
        confidence = float(np.max(pred))
        class_id = int(np.argmax(pred))
        label_encoder = load_joblib_safe(VISION_LABEL_ENCODER_PATH)
        label = label_encoder.inverse_transform([class_id])[0] if label_encoder is not None else str(class_id)
        st.success(f"Predicted visual class / rating band: {label}")
        st.write(f"Confidence: {confidence:.3f}")
    except Exception as exc:
        st.warning(f"Live vision prediction could not run with saved artifacts: {exc}")


def similar_products_section(selected: pd.Series, products: pd.DataFrame) -> None:
    st.subheader("Milestone 3 - Similar Products")
    if products.empty:
        st.warning("Product-level data not found. Run Milestone 0 first.")
        return

    index_df, vectorizer, matrix = build_similarity_index(products)
    query_text = product_text(selected)
    if not query_text:
        st.warning("No product text available for similarity search.")
        return

    query_vec = vectorizer.transform([query_text])
    scores = cosine_similarity(query_vec, matrix).ravel()
    results = index_df.copy()
    results["similarity"] = scores
    results = results[results["product_id"].astype(str) != str(selected.get("product_id"))]
    results = results.sort_values("similarity", ascending=False).head(5)

    for _, item in results.iterrows():
        cols = st.columns([1, 4])
        with cols[0]:
            show_product_image(item)
        with cols[1]:
            st.markdown(f"**{display_value(item.get('product_title'), 'Untitled product')}**")
            st.write(f"Product ID: {item.get('product_id')}")
            st.write(f"Similarity score: {item['similarity']:.3f}")
            price = item.get("price_num")
            st.write("Price: Not available" if pd.isna(price) else f"Price: ${float(price):.2f}")


def product_reviews(product_id: str, reviews: pd.DataFrame | None) -> pd.DataFrame:
    if reviews is None or reviews.empty:
        return pd.DataFrame()
    if "product_id" not in reviews.columns:
        return pd.DataFrame()
    return reviews[reviews["product_id"].astype(str) == str(product_id)].copy()


def review_rating_series(reviews: pd.DataFrame) -> pd.Series:
    for column in ["rating_num", "rating", "overall"]:
        if column in reviews.columns:
            return pd.to_numeric(reviews[column], errors="coerce")
    return pd.Series(np.nan, index=reviews.index)


def review_text_series(reviews: pd.DataFrame) -> pd.Series:
    for column in ["review_text", "text", "review_body", "content"]:
        if column in reviews.columns:
            return reviews[column].fillna("").astype(str)
    return pd.Series("", index=reviews.index, dtype=str)


def summary_section(row: pd.Series, reviews: pd.DataFrame | None) -> None:
    st.subheader("Milestone 5 - Review Pros and Cons Summary")
    pid = str(row.get("product_id"))
    saved_summary = load_csv_safe(REPORTS_DIR / "milestone5_summary_examples.csv")
    if saved_summary is not None and "product_id" in saved_summary.columns:
        match = saved_summary[saved_summary["product_id"].astype(str) == pid]
        if not match.empty:
            first = match.iloc[0]
            st.markdown("**Pros**")
            st.write(display_value(first.get("pred_pros"), "No saved pros summary."))
            st.markdown("**Cons**")
            st.write(display_value(first.get("pred_cons"), "No saved cons summary."))
            return

    selected_reviews = product_reviews(pid, reviews)
    if selected_reviews.empty:
        st.warning("Review-level artifact not found for this product. Run Milestone 0 or Milestone 5 first.")
        return

    st.caption("Lightweight review-based fallback summary")
    ratings = review_rating_series(selected_reviews)
    texts = review_text_series(selected_reviews)
    positive = texts[ratings >= 4]
    negative = texts[ratings <= 2]
    pros = positive[positive.str.len() > 0].map(clean_snippet).head(3).tolist()
    cons = negative[negative.str.len() > 0].map(clean_snippet).head(3).tolist()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Pros**")
        if pros:
            for text in pros:
                st.write(f"- {text}")
        else:
            st.write("No clear positive review snippets found.")
    with col2:
        st.markdown("**Cons**")
        if cons:
            for text in cons:
                st.write(f"- {text}")
        else:
            st.write("No clear negative review snippets found.")


def qa_section(row: pd.Series, reviews: pd.DataFrame | None) -> None:
    st.subheader("Milestone 5 - Grounded Buyer Q&A")
    question = st.text_input("Ask a buyer question", placeholder="Example: Is this good for sensitive skin?")
    if not question:
        return

    selected_reviews = product_reviews(str(row.get("product_id")), reviews)
    if selected_reviews.empty:
        st.warning("No review evidence found for this product.")
        return

    snippets = review_text_series(selected_reviews)
    snippets = snippets[snippets.str.len() > 0].head(80)
    if snippets.empty:
        st.warning("No usable review snippets found for this product.")
        return

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(snippets.tolist() + [question])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    top_idx = np.argsort(scores)[::-1][:3]
    evidence = [(snippets.iloc[i], float(scores[i])) for i in top_idx if scores[i] > 0]
    if not evidence:
        evidence = [(snippets.iloc[0], 0.0)]

    st.info("Extractive fallback answer: based only on retrieved review snippets, the safest answer is to inspect the evidence below. No unsupported facts are added.")
    st.markdown("**Evidence snippets**")
    for text, score in evidence:
        st.write(f"- ({score:.3f}) {clean_snippet(text)}")


def generated_image_section(row: pd.Series) -> None:
    st.subheader("Milestone 6 - Generated Hero Image")
    manifest = load_csv_safe(REPORTS_DIR / "milestone6_generated_image_manifest.csv")
    pid = str(row.get("product_id"))
    if manifest is not None and not manifest.empty and "product_id" in manifest.columns:
        match = manifest[manifest["product_id"].astype(str) == pid]
        if not match.empty:
            image_path = resolve_path(match.iloc[0].get("generated_image_path"))
            if image_path and image_path.exists():
                st.image(image_path, caption="Generated hero image", width="stretch")
                return

    first_image = None
    for directory in GENERATED_IMAGE_DIRS:
        if directory.exists():
            first_image = next(iter(sorted(directory.glob("*.png"))), None)
            if first_image:
                break
    if first_image:
        st.image(first_image, caption="Available generated demo image", width="stretch")
        return

    prompt = f"Product lifestyle photo of {display_value(row.get('product_title'), 'selected product')}, {clean_snippet(product_text(row), 180)}, clean e-commerce style"
    st.warning("Diffusion generation artifact was not found. Run Milestone 6 first.")
    st.code(prompt, language="text")


def sidebar_status(product_data, review_data) -> None:
    st.sidebar.title("Artifact Status")
    tabular_metrics = first_existing(
        [
            PROJECT_ROOT / "reports/milestone1_tabular_mlp_metrics.csv",
            PROJECT_ROOT / "reports/tabular_mlp_model_comparison.csv",
            PROJECT_ROOT / "report/tabular_mlp_model_comparison.csv",
        ]
    )
    vision_metrics = first_existing(
        [
            PROJECT_ROOT / "reports/milestone2_vision_metrics.csv",
            PROJECT_ROOT / "reports/vision_model_comparison.csv",
            PROJECT_ROOT / "report/vision_model_comparison.csv",
        ]
    )
    generated = any(path.exists() and any(path.glob("*.png")) for path in GENERATED_IMAGE_DIRS)
    statuses = {
        "Product data": product_data is not None,
        "Review data": review_data is not None,
        "Tabular model": TABULAR_MODEL_PATH.exists(),
        "Tabular preprocessing artifact": first_existing(TABULAR_PREPROCESSOR_CANDIDATES) is not None,
        "Vision model": first_existing(VISION_MODEL_CANDIDATES) is not None,
        "Embeddings/index": first_existing(EMBEDDING_CANDIDATES) is not None,
        "Generated images": generated,
        "Metrics CSV files": tabular_metrics is not None or vision_metrics is not None,
    }
    for name, ok in statuses.items():
        st.sidebar.write(f"{'OK' if ok else 'Missing'} - {name}")


product_path = first_existing(PRODUCT_LEVEL_CANDIDATES)
products = load_parquet_safe(product_path) if product_path else None
review_path = first_existing(REVIEW_LEVEL_CANDIDATES)
reviews = load_parquet_safe(review_path) if review_path else None
sidebar_status(products, reviews)

st.title("Smart Product Intelligence Demo")
st.caption("Milestone 7 capstone demo for Amazon Reviews 2023 - All_Beauty")

if products is None:
    st.warning("Artifact not found: product-level parquet. Run Milestone 0 first. Showing a placeholder product so the demo UI remains runnable.")
    products = fallback_products()

selected_product = select_product(products)
if selected_product is None:
    st.stop()

product_information(selected_product)
st.divider()
tabular_section(selected_product)
st.divider()
vision_section(selected_product)
st.divider()
similar_products_section(selected_product, products)
st.divider()
summary_section(selected_product, reviews)
st.divider()
qa_section(selected_product, reviews)
st.divider()
generated_image_section(selected_product)
