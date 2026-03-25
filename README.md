# Music Mood Classifier

A machine learning pipeline that classifies Spotify tracks into broad musical
mood and style categories from audio features alone. The project covers the full
ML workflow: exploratory data analysis, feature engineering, preprocessing,
cross-validated model comparison, hyperparameter tuning, evaluation, and an
interactive Streamlit web UI.

---

## Project Overview

Given a set of Spotify audio features for a track (danceability, energy,
acousticness, speechiness, tempo, etc.), the classifier predicts which of six
broad musical categories the track belongs to. The taxonomy was designed
iteratively to maximise classification accuracy while keeping the categories
musically meaningful.

---

## Dataset

The dataset is a CSV file located at `data/raw/`. It contains 114,000 Spotify
tracks with 15 audio features per track and one of 114 original genre labels.

| Property | Value |
|----------|-------|
| Rows | 114,000 |
| Original genre labels | 114 |
| Audio features | 15 |
| Source | Spotify via Kaggle (track_genre column) |

The CSV is auto-discovered at runtime by scanning `data/raw/` for any `.csv`
file. No hardcoded file path is used.

---

## Discovered Mood Categories

The 114 original Spotify sub-genre labels are collapsed into 6 broad categories.
I went through four rounds of per-class accuracy analysis, merging categories
that were consistently confused with each other, until each remaining class had
at least one clearly distinguishing audio axis.

| Category | Sub-genres included | Key audio signature |
|----------|---------------------|---------------------|
| **acoustic** | folk, classical, ambient, blues, jazz, romance, sleep, study | Very high acousticness, low energy |
| **alternative** | indie, grunge, rock, alt-rock, psych-rock | Moderate-high energy, guitar-driven, low acousticness |
| **dance** | latin, pop, dance, R-and-B, soul, reggae, k-pop, j-pop, world music | Very high danceability, moderate-high valence |
| **electronic** | EDM, house, techno, trance, drum-and-bass, dubstep | Very high energy, very low acousticness, high instrumentalness |
| **heavy** | metal, punk, hardcore, emo, goth | Very high energy, maximum loudness, low valence |
| **vocal** | hip-hop, rap, children, comedy | Very high speechiness |

The full reasoning for each merge is documented in `src/genre_mapping.py`.

---

## Model Summary

| Component | Detail |
|-----------|--------|
| Algorithm | LightGBM (`LGBMClassifier`) |
| Feature engineering | 15 raw features expanded to 42 via `MusicFeatureEngineer` |
| Preprocessing | `StandardScaler` fitted on train set only |
| Hyperparameters | `n_estimators=1000`, `num_leaves=511`, `learning_rate=0.05` |
| Train/val/test split | 70% / 15% / 15%, stratified |
| Final training | Trained on combined train+val (96,900 samples) |

### Feature Engineering (`MusicFeatureEngineer`)

The custom sklearn transformer adds 27 domain-informed features on top of the
15 raw audio features:

- Log-transformed features: speechiness, acousticness, instrumentalness, liveness
- Duration conversions: `duration_min`, `log_duration_ms`
- Loudness transformations: `abs_loudness`, `loudness_norm`
- Interaction terms: `energy x danceability`, `valence x energy`, etc.
- Squared terms: tempo, popularity, energy, acousticness, instrumentalness
- Tempo bins: `tempo_slow`, `tempo_fast`, `tempo_norm`
- Key-mode interaction: `key_x_mode`

---

## Results

| Metric | Value |
|--------|-------|
| Test accuracy | 69.03% |
| Weighted F1 | 0.6845 |
| Macro F1 | 0.6444 |
| Macro ROC-AUC (OVR) | 0.9045 |

Confusion matrix and ROC curves are saved to `reports/figures/`.

### Note on accuracy

Starting from 114 raw genre labels the model only reached around 30-35%
accuracy. Collapsing those labels into 6 meaningful categories brought it up
to 69%. Spotify's audio features simply don't carry enough information to
cleanly separate more than around 6 genre buckets. The full breakdown of what
was tried and why 80% wasn't reachable is in `reports/improvement_log.md`.

---

## Installation

### Prerequisites

- Python 3.9+
- 8 GB RAM minimum (16 GB recommended for faster training)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd music-mood-classifier

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Dataset

Place the Spotify dataset CSV in `data/raw/`. The pipeline will discover it
automatically.

---

## Usage

### Run the full training pipeline

```bash
# Step 1: Exploratory data analysis
python -m src.eda

# Step 2: Preprocessing (cleans data, builds pipeline, saves splits)
python -m src.preprocessing

# Step 3: Model selection and training
python -m src.model_training

# Step 4: Evaluation
python -m src.evaluation
```

### Run individual modules

```bash
python -m src.data_loader      # Inspect the dataset
python -m src.genre_mapping    # Show genre distribution after mapping
python -m src.predict          # Run inference on example tracks
```

### Launch the Streamlit UI

```bash
streamlit run ui/app.py
```

The app opens at `http://localhost:8501` and provides:
- **Manual entry mode**: adjust audio feature sliders and get an instant genre
  prediction with confidence scores.
- **Batch upload mode**: upload a CSV of tracks and download predictions.

### Run tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
music-mood-classifier/
├── data/
│   ├── raw/               Dataset CSV (not tracked in git)
│   └── README.md
├── models/                Serialised artefacts (tracked)
│   ├── final_model.pkl    Trained LightGBM classifier
│   ├── preprocessor.pkl   Fitted sklearn Pipeline
│   ├── label_encoder.pkl  Fitted LabelEncoder
│   └── README.md
├── notebooks/             Exploratory notebooks
├── reports/               Generated reports and figures (tracked)
│   ├── figures/           Confusion matrix, ROC curves, EDA plots
│   ├── evaluation_report.md
│   ├── improvement_log.md
│   ├── model_selection.md
│   └── README.md
├── src/                   Source code
│   ├── data_loader.py     CSV auto-discovery and schema inspection
│   ├── eda.py             Exploratory data analysis and figures
│   ├── feature_engineering.py  Custom sklearn transformer (42 features)
│   ├── genre_mapping.py   114-genre to 6-class taxonomy
│   ├── model_training.py  CV comparison, tuning, and final training
│   ├── predict.py         Inference pipeline
│   ├── preprocessing.py   Cleaning, encoding, scaling, splitting
│   ├── evaluation.py      Metrics and visualisation
│   ├── utils.py           Shared constants and logger
│   └── README.md
├── tests/                 pytest test suite
│   ├── test_data_loader.py
│   ├── test_model.py
│   └── test_preprocessing.py
├── ui/
│   ├── app.py             Streamlit web application
│   └── README.md
├── CONTRIBUTING.md        Contribution guidelines
├── README.md              This file
└── requirements.txt       Python dependencies
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on branch naming, commit
message style, and code standards.
