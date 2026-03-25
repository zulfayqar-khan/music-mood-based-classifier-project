"""Streamlit web UI for the Music Mood Classifier.

Provides two input modes:
- Mode 1: manual entry of audio features via sliders and input fields.
- Mode 2: CSV upload for batch genre prediction.

Run with:
    streamlit run ui/app.py
"""

# ==== Standard Library Imports ====
import pickle
import sys
from pathlib import Path

# ==== Third-Party Imports ====
import numpy as np
import pandas as pd
import streamlit as st

# Ensure the project root is on the path when running from any directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ==== Constants ====
MODEL_PATH: Path = PROJECT_ROOT / "models" / "final_model.pkl"
PIPELINE_PATH: Path = PROJECT_ROOT / "models" / "preprocessor.pkl"
ENCODER_PATH: Path = PROJECT_ROOT / "models" / "label_encoder.pkl"

# Audio feature metadata: (display name, min, max, default, step, help_text)
FEATURE_CONFIG = {
    "popularity":        ("Popularity",        0,   100,  50,    1,   "Spotify popularity score (0=obscure, 100=viral)"),
    "duration_ms":       ("Duration (ms)",     0, 600000, 210000, 1000, "Track length in milliseconds"),
    "explicit":          ("Explicit",          0,    1,   0,     1,   "1 if explicit lyrics, 0 otherwise"),
    "danceability":      ("Danceability",      0.0, 1.0,  0.6, 0.01, "How suitable for dancing (0=low, 1=high)"),
    "energy":            ("Energy",            0.0, 1.0,  0.7, 0.01, "Perceptual intensity (0=calm, 1=intense)"),
    "key":               ("Key",               0,   11,   5,    1,   "Musical key (0=C, 1=C#, ..., 11=B)"),
    "loudness":          ("Loudness (dB)",    -60.0, 0.0, -7.0, 0.1, "Average loudness in decibels"),
    "mode":              ("Mode",              0,    1,   1,    1,   "1=major, 0=minor"),
    "speechiness":       ("Speechiness",      0.0, 1.0,  0.05, 0.01, "Presence of spoken words"),
    "acousticness":      ("Acousticness",     0.0, 1.0,  0.2, 0.01, "Confidence the track is acoustic"),
    "instrumentalness":  ("Instrumentalness", 0.0, 1.0,  0.0, 0.01, "Probability of no vocals"),
    "liveness":          ("Liveness",         0.0, 1.0,  0.12, 0.01, "Probability of live recording"),
    "valence":           ("Valence",          0.0, 1.0,  0.5, 0.01, "Musical positiveness (0=sad, 1=happy)"),
    "tempo":             ("Tempo (BPM)",       0.0, 250.0, 120.0, 0.5, "Estimated beats per minute"),
    "time_signature":    ("Time Signature",    1,    7,   4,    1,   "Beats per bar"),
}

TOP_N_PROBA: int = 5
BATCH_MAX_ROWS: int = 500


# ==== Loaders ====

@st.cache_resource(show_spinner="Loading model artefacts...")
def load_artefacts():
    """Load and cache the model, pipeline, and label encoder.

    Returns:
        A tuple of (model, pipeline, label_encoder).
    """
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(PIPELINE_PATH, "rb") as f:
        pipeline = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    return model, pipeline, le


# ==== Helper Functions ====

def predict_single(record: dict, model, pipeline, le) -> tuple:
    """Run inference on a single feature dict.

    Args:
        record: Dict of feature name to value.
        model: Fitted classifier.
        pipeline: Fitted preprocessing pipeline.
        le: Fitted LabelEncoder.

    Returns:
        A tuple of (predicted_label, proba_series) where proba_series is a
        pandas Series of probability scores indexed by genre name, or None if
        the model does not support predict_proba.
    """
    df = pd.DataFrame([record])
    X_scaled = pipeline.transform(df)
    label_int = model.predict(X_scaled)[0]
    genre = le.inverse_transform([label_int])[0]

    proba_series = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
        proba_series = pd.Series(proba, index=le.classes_).sort_values(ascending=False)

    return genre, proba_series


def predict_batch(df_raw: pd.DataFrame, model, pipeline, le) -> pd.DataFrame:
    """Run batch inference on a DataFrame.

    Args:
        df_raw: DataFrame with audio feature columns.
        model: Fitted classifier.
        pipeline: Fitted preprocessing pipeline.
        le: Fitted LabelEncoder.

    Returns:
        df_raw with an appended 'predicted_genre' column.
    """
    feature_cols = list(FEATURE_CONFIG.keys())
    for col in feature_cols:
        if col not in df_raw.columns:
            df_raw[col] = 0
    X = df_raw[feature_cols].copy()
    X_scaled = pipeline.transform(X)
    y_pred = model.predict(X_scaled)
    labels = le.inverse_transform(y_pred)
    result = df_raw.copy()
    result["predicted_genre"] = labels

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)
        top_conf = proba.max(axis=1)
        result["confidence"] = (top_conf * 100).round(1)

    return result


# ==== Page Layout ====

def main() -> None:
    """Main entry point for the Streamlit app."""
    st.set_page_config(
        page_title="Music Mood Classifier",
        page_icon="🎵",
        layout="wide",
    )

    st.title("Music Mood Classifier")
    st.markdown(
        "Predict the **genre** of a track from its Spotify audio features. "
        "Choose an input mode below."
    )

    try:
        model, pipeline, le = load_artefacts()
    except FileNotFoundError as exc:
        st.error(
            f"Model artefact not found: {exc}. "
            "Run the training pipeline first: `python -m src.model_training`"
        )
        return

    st.sidebar.header("Input Mode")
    mode = st.sidebar.radio(
        "Select input mode",
        options=["Manual entry", "Batch CSV upload"],
        index=0,
    )

    if mode == "Manual entry":
        _manual_entry_mode(model, pipeline, le)
    else:
        _batch_upload_mode(model, pipeline, le)


def _manual_entry_mode(model, pipeline, le) -> None:
    """Render the manual feature entry mode.

    Args:
        model: Fitted classifier.
        pipeline: Fitted preprocessing pipeline.
        le: Fitted LabelEncoder.
    """
    st.header("Manual Feature Entry")
    st.markdown("Adjust the sliders to describe a track, then click **Predict**.")

    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]
    record = {}
    feature_keys = list(FEATURE_CONFIG.keys())

    for idx, key in enumerate(feature_keys):
        label, lo, hi, default, step, help_text = FEATURE_CONFIG[key]
        col = columns[idx % 3]
        with col:
            if isinstance(step, int):
                record[key] = st.slider(
                    label, min_value=int(lo), max_value=int(hi),
                    value=int(default), step=step, help=help_text,
                )
            else:
                record[key] = st.slider(
                    label, min_value=float(lo), max_value=float(hi),
                    value=float(default), step=float(step), help=help_text,
                )

    st.markdown("---")
    if st.button("Predict Genre", type="primary"):
        genre, proba_series = predict_single(record, model, pipeline, le)

        st.success(f"Predicted genre: **{genre}**")

        if proba_series is not None:
            st.subheader(f"Top {TOP_N_PROBA} confidence scores")
            top = proba_series.head(TOP_N_PROBA)
            chart_df = top.reset_index()
            chart_df.columns = ["Genre", "Probability"]
            st.bar_chart(chart_df.set_index("Genre"))

            with st.expander("All genre probabilities"):
                display_df = proba_series.reset_index()
                display_df.columns = ["Genre", "Probability"]
                display_df["Probability (%)"] = (display_df["Probability"] * 100).round(2)
                st.dataframe(display_df[["Genre", "Probability (%)"]], use_container_width=True)


def _batch_upload_mode(model, pipeline, le) -> None:
    """Render the batch CSV upload mode.

    Args:
        model: Fitted classifier.
        pipeline: Fitted preprocessing pipeline.
        le: Fitted LabelEncoder.
    """
    st.header("Batch CSV Upload")
    st.markdown(
        f"Upload a CSV file with up to **{BATCH_MAX_ROWS} rows**. "
        "The file must contain the audio feature columns listed in `data/README.md`. "
        "Any missing feature columns will be filled with zeros."
    )

    uploaded = st.file_uploader("Upload CSV", type="csv")

    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read the uploaded file: {exc}")
            return

        if len(df_raw) > BATCH_MAX_ROWS:
            st.warning(f"File has {len(df_raw)} rows. Only the first {BATCH_MAX_ROWS} will be processed.")
            df_raw = df_raw.head(BATCH_MAX_ROWS)

        st.markdown(f"**{len(df_raw)} rows** detected. Running predictions...")

        with st.spinner("Predicting..."):
            result_df = predict_batch(df_raw, model, pipeline, le)

        st.success("Predictions complete.")
        st.dataframe(result_df, use_container_width=True)

        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download predictions as CSV",
            data=csv_bytes,
            file_name="genre_predictions.csv",
            mime="text/csv",
        )


# ==== Entry Point ====

if __name__ == "__main__":
    main()
