# An AI-Based Hybrid Music Recommendation System

*COMP1682 Final Year Individual Project*
Manuela Geshtenja — University of Greenwich, April 2026

A lightweight hybrid music recommendation system combining Collaborative Filtering (CF)
and Content-Based Filtering (CBF) using a weighted hybrid approach with fallback
switching, implemented with a Streamlit web interface.

---

## Project Structure

```
project/
    app.py                       Streamlit frontend
    hybrid_recommender.py        Core recommendation logic (CF, CBF, hybrid, fallback)
    evaluate.py                  Full-population offline evaluation script
    evaluate_within_coverage.py  Within-coverage evaluation (CF∩CBF seeds only)
    archive/
        gui.py                   Original command-line prototype (superseded by app.py)
    data/
        spotify_dataset.csv      Spotify playlist dataset (CF input)
        dataset.csv              Spotify audio features dataset (CBF input)
    requirements.txt             Python dependencies
    README.md                    This file
```

---

## Requirements

- Python 3.8 or higher
- The two datasets in the `data/` folder (see [Datasets](#datasets) below)

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

**First launch:** the CF and CBF similarity matrices are built from scratch, which
takes a few minutes on a standard laptop. The result is cached automatically — all
subsequent interactions respond in under 3 seconds.

---

## Running the Evaluation

**Full-population evaluation** (200 randomly sampled users, leave-one-out):

```bash
python evaluate.py
```

**Within-coverage evaluation** (seeds where both CF and CBF models have coverage):

```bash
python evaluate_within_coverage.py
```

Both scripts print Precision@10, Recall@10, NDCG@10, and Intra-List Diversity for
each model, along with recommendation mode coverage statistics.

---

## Datasets

Two publicly available Kaggle datasets are required. Place both files inside the
`data/` folder before running the app.

| Dataset | Source | File |
|---|---|---|
| Spotify Playlists (CF) | [Larxel, Kaggle 2020](https://www.kaggle.com/datasets/andrewmvd/spotify-playlists) | `spotify_dataset.csv` |
| Spotify Tracks Dataset (CBF) | [Maharshi Pandya, Kaggle 2022](https://www.kaggle.com/dsv/4372070) | `dataset.csv` |

---

## How It Works

The system uses a **weighted hybrid** as its primary method, with **fallback switching**
when one model lacks coverage for a given song:

| Mode | Condition | Method |
|---|---|---|
| **Hybrid** | Song in both CF and CBF models | `0.6 × CF_score + 0.4 × CBF_score` |
| **CF only** | Song in CF model only | Item-based cosine similarity over playlist co-occurrence |
| **CBF only** | Song in CBF model only | Cosine similarity over 9 standardised audio features |
| **Popularity fallback** | Song in neither model | Top-n most frequently occurring songs in playlist data |

The interface always shows which mode was used for each recommendation.

---

## Key Results

Full-population evaluation (200 users, leave-one-out):

| Model | Precision@10 | NDCG@10 | ILD |
|---|---|---|---|
| Random baseline | 0.0005 | 0.0022 | 0.0037 |
| CF only | 0.0045 | 0.0272 | 0.0003 |
| CBF only | 0.0000 | 0.0000 | 0.0003 |
| **Hybrid (proposed)** | **0.0035** | **0.0209** | **0.0003** |

Within-coverage evaluation (restricted to seeds present in both CF and CBF models, n = 200):

| Model | Precision@10 | NDCG@10 |
|---|---|---|
| CF only | 0.1950 | 0.0980 |
| **Hybrid** | **0.2250** | **0.1303** |

A paired t-test on per-user ILD distributions confirmed a statistically significant
diversity advantage for the hybrid approach (p = 0.019).

---

## Libraries Used

- `pandas`, `numpy` — data handling
- `scikit-learn` — cosine similarity, StandardScaler
- `streamlit` — web interface
- `plotly` — radar charts and visualisations

All dependencies are listed in `requirements.txt`.

---

## Notes

- No dedicated recommendation system framework is used. All CF and CBF logic is
  implemented manually using `pandas` and `scikit-learn`.
- The `archive/gui.py` file is the original command-line prototype retained for
  reference. It is not used by the current system.
- The true overlap between datasets is limited due to reliance on string-based
  matching (`artist - track`), and is likely underestimated due to naming
  inconsistencies between the two sources.
