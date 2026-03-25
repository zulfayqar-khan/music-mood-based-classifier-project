# UI (`ui/`)

A Streamlit web application for interactive music genre prediction.

## Dependencies

All dependencies are listed in the root `requirements.txt`. The UI specifically requires:

- `streamlit>=1.29.0`
- `pandas`
- `numpy`

## Running the App

From the project root directory:

```bash
streamlit run ui/app.py
```

The app opens in your browser at `http://localhost:8501`.

## Input Modes

### Mode 1: Manual Entry

Adjust sliders for each of the 15 Spotify audio features (popularity, danceability,
energy, etc.) and click **Predict Genre** to see:

- The predicted super-genre label.
- A bar chart of the confidence scores for all 6 super-genres.
- An expandable table of all genre probabilities.

### Mode 2: Batch CSV Upload

Upload a CSV file containing audio feature columns. The app:

1. Accepts up to 500 rows per upload.
2. Fills any missing feature columns with zeros.
3. Returns a table with a `predicted_genre` column appended.
4. Offers a download button for the results CSV.

## Notes

- The app loads `models/final_model.pkl`, `models/preprocessor.pkl`, and
  `models/label_encoder.pkl`. Run the training pipeline first if these files
  do not exist.
- Model artefacts are cached via `@st.cache_resource` so they are only loaded
  once per session.
