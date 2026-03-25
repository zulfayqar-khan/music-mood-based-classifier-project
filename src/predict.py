"""Inference pipeline for new inputs using the trained model and preprocessor."""

# ==== Standard Library Imports ====
from pathlib import Path
from typing import Dict, List, Optional, Union

# ==== Third-Party Imports ====
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ==== Internal Imports ====
from src.preprocessing import load_pipeline
from src.model_training import load_model
from src.utils import NUMERIC_FEATURES, get_logger

# ==== Module Logger ====
logger = get_logger(__name__)

# ==== Constants ====
# Columns that are dropped before inference (metadata, not audio features).
INFERENCE_DROP_COLUMNS: List[str] = ["track_id", "artists", "album_name", "track_name"]


# ==== Inference Functions ====

def load_artifacts(
    model_path: Optional[Path] = None,
    pipeline_path: Optional[Path] = None,
) -> tuple:
    """Load the trained model and preprocessing pipeline from disk.

    Args:
        model_path: Optional path to the model pickle. Defaults to models/final_model.pkl.
        pipeline_path: Optional path to the preprocessor pickle. Defaults to models/preprocessor.pkl.

    Returns:
        A tuple of (model, pipeline).
    """
    model = load_model(model_path)
    pipeline = load_pipeline(pipeline_path)
    return model, pipeline


def preprocess_input(
    data: Union[pd.DataFrame, Dict],
    pipeline,
) -> np.ndarray:
    """Prepare raw input data for inference.

    Accepts either a single record dict or a DataFrame. Drops metadata columns,
    fills missing numeric features with zero, and applies the fitted pipeline.

    Args:
        data: A single record as a dict or a DataFrame of records.
        pipeline: The fitted sklearn preprocessing Pipeline.

    Returns:
        A scaled numpy array ready for model.predict().
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    # Drop metadata columns if present.
    drop_cols = [c for c in INFERENCE_DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Keep only the expected feature columns, inserting zeros for any missing.
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0

    df = df[NUMERIC_FEATURES].copy()

    # Cast the explicit column to int if still boolean.
    if df["explicit"].dtype == bool:
        df["explicit"] = df["explicit"].astype(int)

    X_scaled = pipeline.transform(df)
    return X_scaled


def predict(
    data: Union[pd.DataFrame, Dict],
    label_encoder: Optional[LabelEncoder] = None,
    model=None,
    pipeline=None,
) -> List[str]:
    """Predict mood/genre labels for one or more input records.

    Args:
        data: A single record as a dict or a DataFrame of records.
        label_encoder: Optional fitted LabelEncoder to decode integer predictions
            back to string labels. If None, integer labels are returned.
        model: Optional pre-loaded model. Loaded from disk if not provided.
        pipeline: Optional pre-loaded pipeline. Loaded from disk if not provided.

    Returns:
        List of predicted class label strings (or integers if label_encoder is None).
    """
    if model is None or pipeline is None:
        model, pipeline = load_artifacts()

    X_scaled = preprocess_input(data, pipeline)
    y_pred = model.predict(X_scaled)

    if label_encoder is not None:
        return label_encoder.inverse_transform(y_pred).tolist()
    return y_pred.tolist()


def predict_proba(
    data: Union[pd.DataFrame, Dict],
    label_encoder: Optional[LabelEncoder] = None,
    model=None,
    pipeline=None,
) -> pd.DataFrame:
    """Return probability scores for each class for one or more input records.

    Args:
        data: A single record as a dict or a DataFrame of records.
        label_encoder: Optional fitted LabelEncoder used to name the columns.
        model: Optional pre-loaded model. Loaded from disk if not provided.
        pipeline: Optional pre-loaded pipeline. Loaded from disk if not provided.

    Returns:
        DataFrame with one row per input and one column per class, containing
        the predicted probability scores.

    Raises:
        AttributeError: If the loaded model does not support predict_proba.
    """
    if model is None or pipeline is None:
        model, pipeline = load_artifacts()

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            f"The model ({type(model).__name__}) does not support predict_proba. "
            "Use predict() instead."
        )

    X_scaled = preprocess_input(data, pipeline)
    proba = model.predict_proba(X_scaled)

    col_names = (
        label_encoder.classes_.tolist()
        if label_encoder is not None
        else [str(i) for i in range(proba.shape[1])]
    )
    return pd.DataFrame(proba, columns=col_names)


# ==== Entry Point ====

if __name__ == "__main__":
    from src.data_loader import load_data, discover_target
    from src.preprocessing import (
        handle_missing_values,
        encode_features,
        detect_and_clip_outliers,
        prepare_features_target,
        encode_target,
        split_data,
        load_pipeline,
    )

    # Load a few test samples for demonstration.
    raw_df = load_data()
    target_col = discover_target(raw_df)
    df_clean = handle_missing_values(raw_df)
    df_enc = encode_features(df_clean)
    df_clip = detect_and_clip_outliers(df_enc)
    X, y_raw = prepare_features_target(df_clip, target_col)
    _, le = encode_target(y_raw)
    _, _, X_test, _, _, y_test = split_data(X, encode_target(y_raw)[0])

    pipeline = load_pipeline()
    model, _ = load_artifacts()

    sample = X_test.iloc[:5]
    predictions = predict(sample, label_encoder=le, model=model, pipeline=pipeline)
    print("Predicted labels:", predictions)

    if hasattr(model, "predict_proba"):
        proba_df = predict_proba(
            sample, label_encoder=le, model=model, pipeline=pipeline
        )
        print("Top-3 probabilities per sample:")
        for i, row in proba_df.iterrows():
            top3 = row.nlargest(3)
            print(f"  Sample {i}: {top3.to_dict()}")
