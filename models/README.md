# Models Directory

This directory stores all serialised model artefacts produced during training.

## Files

| File | Description |
|------|-------------|
| `preprocessor.pkl` | Fitted sklearn Pipeline: `MusicFeatureEngineer` followed by `StandardScaler`. Apply to any raw feature DataFrame before prediction. |
| `final_model.pkl` | The final trained classifier (LightGBM). Load and call `.predict()` or `.predict_proba()`. |
| `label_encoder.pkl` | Fitted `sklearn.preprocessing.LabelEncoder`. Maps integer predictions back to genre name strings. |

## How to Load and Use

```python
import pickle
import numpy as np
import pandas as pd

# Load artefacts.
with open("models/preprocessor.pkl", "rb") as f:
    pipeline = pickle.load(f)

with open("models/final_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Prepare a new record (must have all 15 audio feature columns).
record = pd.DataFrame([{
    "popularity": 65, "duration_ms": 210000, "explicit": 0,
    "danceability": 0.7, "energy": 0.8, "key": 5,
    "loudness": -5.2, "mode": 1, "speechiness": 0.06,
    "acousticness": 0.1, "instrumentalness": 0.0,
    "liveness": 0.12, "valence": 0.75, "tempo": 128.0,
    "time_signature": 4,
}])

# Preprocess and predict.
X_scaled = pipeline.transform(record)
label_int = model.predict(X_scaled)[0]
genre     = le.inverse_transform([label_int])[0]
print(f"Predicted genre: {genre}")

# Probability scores.
proba = model.predict_proba(X_scaled)[0]
top3  = sorted(zip(le.classes_, proba), key=lambda x: -x[1])[:3]
for genre_name, prob in top3:
    print(f"  {genre_name}: {prob:.3f}")
```

## Notes

- The `preprocessor.pkl` pipeline applies feature engineering (adding 27 derived
  features to the original 15) and then StandardScaler, producing a 42-dimensional
  scaled array.
- The `final_model.pkl` expects this 42-dimensional scaled input.
- Do not apply scaling manually before calling `pipeline.transform()`.
- Raw data files (`data/raw/`) are not committed. Trained model files are tracked.
