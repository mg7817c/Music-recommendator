"""
Within-coverage evaluation: restricts to users whose seed AND test song
are both in the shared CF+CBF index (16,853 songs).
This gives the hybrid a completely fair test with both models contributing.
The Streamlit app still uses fallback for wider coverage.

Run: python evaluate_within_coverage.py
"""

import os
import pickle
import time
import math
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from hybrid_recommender import initialise_recommender, _normalise, ALPHA_CF, ALPHA_CBF

CACHE_PATH   = "recommender_cache.pkl"
MAX_USERS    = 200
SEED         = 42
K_VALUES     = [10, 20]
FEATURE_COLS = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "loudness", "speechiness", "tempo", "valence",
]


def load_state():
    if os.path.exists(CACHE_PATH):
        print("Loading from cache...")
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    print("Building from scratch (slow, one time only)...")
    state = initialise_recommender()
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(state, f)
    print("Saved to cache.")
    return state


def build_audio_lookup(audio_data):
    usable = [c for c in FEATURE_COLS if c in audio_data.columns]
    lookup = (audio_data.drop_duplicates("song_id")
              [["song_id"] + usable]
              .dropna(subset=usable)
              .set_index("song_id"))
    return lookup, usable


def precision_at_k(recs, relevant, k):
    return 1.0 if relevant in recs[:k] else 0.0

def recall_at_k(recs, relevant, k):
    return 1.0 if relevant in recs[:k] else 0.0

def ndcg_at_k(recs, relevant, k):
    if relevant not in recs[:k]:
        return 0.0
    return 1.0 / math.log2(recs[:k].index(relevant) + 2)

def ild_at_k(recs, audio_lookup, usable_cols, k):
    subset = audio_lookup.reindex(recs[:k]).dropna(subset=usable_cols)
    if len(subset) < 2:
        return 0.0
    mat = subset[usable_cols].to_numpy()
    sim = sk_cosine(mat)
    n   = len(mat)
    d   = [1 - sim[i][j] for i in range(n) for j in range(i+1, n)]
    return float(np.mean(d)) if d else 0.0


def get_hybrid(cf_df, cbf_df, seed, topn):
    cf_s   = cf_df[seed].drop(labels=[seed], errors="ignore")
    cbf_s  = cbf_df[seed].drop(labels=[seed], errors="ignore")
    union  = cf_s.index.union(cbf_s.index)
    scores = (ALPHA_CF  * _normalise(cf_s.reindex(union, fill_value=0.0)) +
              ALPHA_CBF * _normalise(cbf_s.reindex(union, fill_value=0.0)))
    return scores.drop(labels=[seed], errors="ignore").nlargest(topn).index.tolist()

def get_cf(cf_df, seed, topn):
    return cf_df[seed].drop(labels=[seed], errors="ignore").nlargest(topn).index.tolist()

def get_cbf(cbf_df, seed, topn):
    return cbf_df[seed].drop(labels=[seed], errors="ignore").nlargest(topn).index.tolist()


if __name__ == "__main__":
    t0    = time.time()
    state = load_state()

    cf_df  = state["cf_similarity_df"]
    cbf_df = state["cbf_similarity_df"]

    shared = set(cf_df.index) & set(cbf_df.index)
    print(f"Shared CF+CBF songs: {len(shared)}")

    audio_lookup, usable_cols = build_audio_lookup(state["audio_data"])

    user_to_songs = (
        state["playlist_data"]
        .groupby("user")["song_id"]
        .apply(list)
        .to_dict()
    )

    # filter: seed AND test song both in shared index
    eligible = [
        user for user, songs in user_to_songs.items()
        if len(songs) >= 5
        and songs[-2] in shared
        and songs[-1] in shared
    ]
    print(f"Eligible users (seed+test in shared index): {len(eligible)}")

    rng   = np.random.default_rng(SEED)
    users = (rng.choice(eligible, size=MAX_USERS, replace=False).tolist()
             if len(eligible) > MAX_USERS else eligible)
    print(f"Evaluating {len(users)} users...\n")

    all_songs = state["playlist_data"]["song_id"].unique().tolist()
    max_k     = max(K_VALUES)

    metrics = {m: {k: {"precision": [], "recall": [], "ndcg": [], "ild": []}
                   for k in K_VALUES}
               for m in ["random", "cf", "cbf", "hybrid"]}

    for i, user in enumerate(users):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(users)}")

        songs     = user_to_songs[user]
        seed_song = songs[-2]
        test_song = songs[-1]

        random_ids = rng.choice(all_songs, size=max_k, replace=False).tolist()
        cf_ids     = get_cf(cf_df, seed_song, max_k)
        cbf_ids    = get_cbf(cbf_df, seed_song, max_k)
        hybrid_ids = get_hybrid(cf_df, cbf_df, seed_song, max_k)

        for model, recs in [("random", random_ids), ("cf", cf_ids),
                             ("cbf", cbf_ids), ("hybrid", hybrid_ids)]:
            for k in K_VALUES:
                metrics[model][k]["precision"].append(precision_at_k(recs, test_song, k))
                metrics[model][k]["recall"].append(recall_at_k(recs, test_song, k))
                metrics[model][k]["ndcg"].append(ndcg_at_k(recs, test_song, k))
                metrics[model][k]["ild"].append(ild_at_k(recs, audio_lookup, usable_cols, k))

    print(f"\nWithin-coverage evaluation (seed + test both in CF \u2229 CBF index)")
    print(f"N users: {len(users)}\n")

    for k in K_VALUES:
        print(f"--- k={k} ---")
        print(f"{'Model':<12} {'Precision':>10} {'Recall':>10} {'NDCG':>10} {'ILD':>10}")
        print("-" * 56)
        for model in ["random", "cf", "cbf", "hybrid"]:
            p   = np.mean(metrics[model][k]["precision"])
            r   = np.mean(metrics[model][k]["recall"])
            nd  = np.mean(metrics[model][k]["ndcg"])
            ild = np.mean(metrics[model][k]["ild"])
            print(f"{model:<12} {p:>10.4f} {r:>10.4f} {nd:>10.4f} {ild:>10.4f}")
        t, p_val = stats.ttest_rel(
            metrics["hybrid"][k]["ild"], metrics["cf"][k]["ild"])
        print(f"Hybrid vs CF ILD: t={t:.3f}, p={p_val:.3f}\n")

    print(f"Total time: {time.time()-t0:.1f}s")