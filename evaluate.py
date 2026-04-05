import math
import random
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from scipy import stats

from hybrid_recommender import (
    initialise_recommender,
    get_cf_recommendations,
    get_cbf_recommendations,
    get_hybrid_recommendations,
    get_popular_fallback,
)

K           = 10
NUM_USERS   = 200
MIN_SONGS   = 5
RANDOM_SEED = 42
CACHE_PATH  = "recommender_cache.pkl"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_state():
    if os.path.exists(CACHE_PATH):
        print("Loading from cache (fast)...")
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    print("Building from scratch (slow, one time only)...")
    state = initialise_recommender()
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(state, f)
    print("Saved to cache.")
    return state


def precision_at_k(recommended, relevant, k):
    return sum(1 for r in recommended[:k] if r in relevant) / k


def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0
    return sum(1 for r in recommended[:k] if r in relevant) / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    top_k = recommended[:k]
    dcg   = sum(1 / math.log2(i + 2) for i, r in enumerate(top_k) if r in relevant)
    idcg  = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def intra_list_diversity(recommended, audio_data, feature_cols):
    rows      = audio_data[audio_data["song_id"].isin(recommended)]
    available = [c for c in feature_cols if c in rows.columns]
    if len(rows) < 2 or not available:
        return 0.0
    features = rows[available].dropna()
    if len(features) < 2:
        return 0.0
    sim_matrix = cosine_similarity(features)
    n = len(sim_matrix)
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1 - sim_matrix[i][j]
            count += 1
    return total / count if count > 0 else 0.0


def build_test_set(playlist_data, min_songs, num_users):
    user_songs = defaultdict(list)
    for _, row in playlist_data.iterrows():
        user_songs[row["user"]].append(row["song_id"])
    eligible = {u: songs for u, songs in user_songs.items()
                if len(set(songs)) >= min_songs}
    sampled = random.sample(list(eligible.keys()), min(num_users, len(eligible)))
    test_set = {}
    for user in sampled:
        songs = list(dict.fromkeys(eligible[user]))
        test_set[user] = {"train": songs[:-1], "test": songs[-1]}
    return test_set


def get_recs_for_song(seed_id, model, state, k):
    cf  = state["cf_similarity_df"]
    cbf = state["cbf_similarity_df"]
    pop = state["popularity_df"]

    if model == "random":
        candidates = [s for s in state["catalog"]["song_id"] if s != seed_id]
        return random.sample(candidates, min(k, len(candidates)))
    elif model == "cf":
        recs = get_cf_recommendations(seed_id, cf, n=k)
        return list(recs.index) if len(recs) > 0 else list(get_popular_fallback(pop, seed_id, k)["song_id"])
    elif model == "cbf":
        recs = get_cbf_recommendations(seed_id, cbf, n=k)
        return list(recs.index) if len(recs) > 0 else list(get_popular_fallback(pop, seed_id, k)["song_id"])
    elif model == "hybrid":
        in_cf, in_cbf = seed_id in cf.index, seed_id in cbf.index
        if in_cf and in_cbf:
            recs = get_hybrid_recommendations(seed_id, cf, cbf, n=k)
        elif in_cf:
            recs = get_cf_recommendations(seed_id, cf, n=k)
        elif in_cbf:
            recs = get_cbf_recommendations(seed_id, cbf, n=k)
        else:
            return list(get_popular_fallback(pop, seed_id, k)["song_id"])
        return list(recs.index) if len(recs) > 0 else list(get_popular_fallback(pop, seed_id, k)["song_id"])
    return []


def evaluate():
    state = load_state()
    playlist_data = state["playlist_data"]
    audio_data    = state["audio_data"]

    cf_songs  = set(state["cf_similarity_df"].index)
    cbf_songs = set(state["cbf_similarity_df"].index)
    shared    = cf_songs & cbf_songs
    print(f"\nCF: {len(cf_songs)} | CBF: {len(cbf_songs)} | Shared: {len(shared)} ({len(shared)/len(cf_songs)*100:.1f}%)\n")

    feature_cols = ["energy","valence","tempo","danceability",
                    "acousticness","instrumentalness","liveness","loudness","speechiness"]

    print(f"Building test set ({NUM_USERS} users, min {MIN_SONGS} songs)...")
    test_set = build_test_set(playlist_data, MIN_SONGS, NUM_USERS)
    print(f"Evaluating {len(test_set)} users...")

    models  = ["random", "cf", "cbf", "hybrid"]
    results = {m: {"precision": [], "recall": [], "ndcg": [], "ild": []} for m in models}

    for idx, (user, data) in enumerate(test_set.items()):
        if idx % 50 == 0:
            print(f"  {idx}/{len(test_set)} done...")
        seed_id   = data["train"][-1]
        relevant  = {data["test"]}
        for model in models:
            recs = get_recs_for_song(seed_id, model, state, K)
            if not recs:
                continue
            results[model]["precision"].append(precision_at_k(recs, relevant, K))
            results[model]["recall"].append(recall_at_k(recs, relevant, K))
            results[model]["ndcg"].append(ndcg_at_k(recs, relevant, K))
            results[model]["ild"].append(intra_list_diversity(recs, audio_data, feature_cols))

    print("\n" + "="*60)
    print(f"RESULTS  (k={K}, n={len(test_set)} users)")
    print("="*60)
    labels = {"random":"Random baseline","cf":"CF only","cbf":"CBF only","hybrid":"Hybrid (proposed)"}
    rows = []
    for model in models:
        r = results[model]
        if not r["precision"]: continue
        row = {"Model": labels[model],
               f"Precision@{K}": round(np.mean(r["precision"]),4),
               f"Recall@{K}":    round(np.mean(r["recall"]),4),
               f"NDCG@{K}":      round(np.mean(r["ndcg"]),4),
               "ILD":            round(np.mean(r["ild"]),4)}
        rows.append(row)
        print(f"\n{row['Model']}\n  Precision@{K}: {row[f'Precision@{K}']}\n  Recall@{K}:    {row[f'Recall@{K}']}\n  NDCG@{K}:      {row[f'NDCG@{K}']}\n  ILD:           {row['ILD']}")

    print("\n" + "="*60)
    pd.DataFrame(rows).to_csv("evaluation_results.csv", index=False)
    print("Saved to evaluation_results.csv")

    for key, name in [("precision",f"Precision@{K}"),("ndcg",f"NDCG@{K}"),("ild","ILD")]:
        h, c = results["hybrid"][key], results["cf"][key]
        if len(h) == len(c) and len(h) > 1:
            _, p = stats.ttest_rel(h, c)
            print(f"Hybrid vs CF - {name}: p={p:.4f} ({'significant' if p < 0.05 else 'not significant'})")

    mode_counts = {"HYBRID": 0, "CF": 0, "CBF": 0, "POPULAR": 0}
    for user, data in test_set.items():
        seed_id = data["train"][-1]
        in_cf  = seed_id in state["cf_similarity_df"].index
        in_cbf = seed_id in state["cbf_similarity_df"].index
        if in_cf and in_cbf:   mode_counts["HYBRID"] += 1
        elif in_cf:            mode_counts["CF"] += 1
        elif in_cbf:           mode_counts["CBF"] += 1
        else:                  mode_counts["POPULAR"] += 1

    total = len(test_set)
    print("\nRecommendation mode coverage:")
    for mode, count in mode_counts.items():
        print(f"  {mode}: {count}/{total} ({count/total*100:.1f}%)")


if __name__ == "__main__":
    evaluate()