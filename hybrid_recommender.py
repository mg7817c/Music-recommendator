import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# CONFIG
PLAYLIST_PATH = "data/spotify_dataset.csv"
AUDIO_PATH = "data/dataset.csv"

CF_MAX_SONGS = 5000
CBF_MAX_SONGS = 5000

ALPHA_CF = 0.6
ALPHA_CBF = 0.4
TOP_N = 10

VERBOSE = False


def log(*args):
    if VERBOSE:
        print(*args)


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()
    text = text.replace("[", "").replace("]", "").replace("'", "").replace('"', "")

    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\[.*?\]", "", text)

    text = re.sub(r"\bfeat\b.*", "", text)
    text = re.sub(r"\bft\b.*", "", text)
    text = re.sub(r"\bfeaturing\b.*", "", text)

    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_datasets():
    playlist_data = pd.read_csv(
        PLAYLIST_PATH,
        engine="python",
        on_bad_lines="skip"
    )

    audio_data = pd.read_csv(
        AUDIO_PATH,
        engine="python",
        on_bad_lines="skip"
    )

    playlist_data.columns = (
        playlist_data.columns
        .str.replace('"', "", regex=False)
        .str.strip()
        .str.lower()
    )

    audio_data.columns = (
        audio_data.columns
        .str.replace('"', "", regex=False)
        .str.strip()
        .str.lower()
    )

    playlist_data = playlist_data.rename(columns={
        "user_id": "user",
        "artistname": "artist",
        "trackname": "track",
        "playlistname": "playlist"
    })

    audio_data = audio_data.rename(columns={
        "artists": "artist",
        "track_name": "track"
    })

    playlist_data["artist_clean"] = playlist_data["artist"].apply(clean_text)
    playlist_data["track_clean"] = playlist_data["track"].apply(clean_text)

    audio_data["artist_clean"] = audio_data["artist"].apply(clean_text)
    audio_data["track_clean"] = audio_data["track"].apply(clean_text)

    playlist_data = playlist_data[
        (playlist_data["artist_clean"] != "") &
        (playlist_data["track_clean"] != "")
    ].copy()

    audio_data = audio_data[
        (audio_data["artist_clean"] != "") &
        (audio_data["track_clean"] != "")
    ].copy()

    playlist_data["song_id"] = (
        playlist_data["artist_clean"] + " - " + playlist_data["track_clean"]
    )
    audio_data["song_id"] = (
        audio_data["artist_clean"] + " - " + audio_data["track_clean"]
    )

    return playlist_data, audio_data


def find_overlap(playlist_data, audio_data):
    playlist_songs = set(playlist_data["song_id"].unique())
    audio_songs = set(audio_data["song_id"].unique())

    shared_songs = playlist_songs.intersection(audio_songs)
    playlist_only = playlist_songs - shared_songs
    audio_only = audio_songs - shared_songs

    log("Unique playlist songs:", len(playlist_songs))
    log("Unique audio songs:", len(audio_songs))
    log("Shared songs:", len(shared_songs))

    return shared_songs, playlist_only, audio_only


def build_cf_model(playlist_data):
    top_songs = playlist_data["song_id"].value_counts().head(CF_MAX_SONGS).index
    playlist_small = playlist_data[playlist_data["song_id"].isin(top_songs)].copy()

    user_song_matrix = pd.crosstab(
        playlist_small["user"],
        playlist_small["song_id"]
    )

    cf_similarity = cosine_similarity(user_song_matrix.T)

    cf_similarity_df = pd.DataFrame(
        cf_similarity,
        index=user_song_matrix.columns,
        columns=user_song_matrix.columns
    )

    return cf_similarity_df


def build_cbf_model(audio_data):
    song_features = audio_data.drop_duplicates(subset=["song_id"]).copy()

    feature_cols = [
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "liveness",
        "loudness",
        "speechiness",
        "tempo",
        "valence"
    ]

    missing_cols = [col for col in feature_cols if col not in song_features.columns]
    if missing_cols:
        raise KeyError(f"Missing audio feature columns in audio dataset: {missing_cols}")

    song_features = song_features.head(CBF_MAX_SONGS).copy()

    X = song_features[feature_cols].dropna().copy()
    song_features = song_features.loc[X.index].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cbf_similarity = cosine_similarity(X_scaled)

    cbf_similarity_df = pd.DataFrame(
        cbf_similarity,
        index=song_features["song_id"],
        columns=song_features["song_id"]
    )

    return cbf_similarity_df


def build_popularity_table(playlist_data):
    popularity = (
        playlist_data["song_id"]
        .value_counts()
        .reset_index()
    )
    popularity.columns = ["song_id", "count"]
    return popularity


def get_cf_recommendations(song_id, cf_similarity_df, n=10):
    if song_id not in cf_similarity_df.index:
        return pd.Series(dtype=float)

    scores = cf_similarity_df[song_id].sort_values(ascending=False)
    scores = scores.drop(labels=[song_id], errors="ignore")
    return scores.head(n)


def get_cbf_recommendations(song_id, cbf_similarity_df, n=10):
    if song_id not in cbf_similarity_df.index:
        return pd.Series(dtype=float)

    scores = cbf_similarity_df[song_id].sort_values(ascending=False)
    scores = scores.drop(labels=[song_id], errors="ignore")
    return scores.head(n)


def get_hybrid_recommendations(song_id, cf_similarity_df, cbf_similarity_df, n=10):
    if song_id not in cf_similarity_df.index or song_id not in cbf_similarity_df.index:
        return pd.Series(dtype=float)

    cf_scores = cf_similarity_df[song_id]
    cbf_scores = cbf_similarity_df[song_id]

    common_index = cf_scores.index.intersection(cbf_scores.index)

    hybrid_scores = (
        ALPHA_CF * cf_scores.loc[common_index] +
        ALPHA_CBF * cbf_scores.loc[common_index]
    )

    hybrid_scores = hybrid_scores.sort_values(ascending=False)
    hybrid_scores = hybrid_scores.drop(labels=[song_id], errors="ignore")

    return hybrid_scores.head(n)


def get_popular_fallback(popularity_df, exclude_song_id=None, n=10):
    df = popularity_df.copy()
    if exclude_song_id is not None:
        df = df[df["song_id"] != exclude_song_id]
    return df.head(n)


def recommend_from_song_id(
    song_id,
    shared_songs,
    playlist_only,
    audio_only,
    cf_similarity_df,
    cbf_similarity_df,
    popularity_df,
    n=10
):
    in_shared = song_id in shared_songs
    in_cf = song_id in cf_similarity_df.index
    in_cbf = song_id in cbf_similarity_df.index

    print("\nSelected song:", song_id)
    print("In shared_songs:", in_shared)
    print("In CF model:", in_cf)
    print("In CBF model:", in_cbf)

    if in_cf and in_cbf:
        recs = get_hybrid_recommendations(song_id, cf_similarity_df, cbf_similarity_df, n=n)
        if len(recs) == 0:
            fallback = get_popular_fallback(popularity_df, exclude_song_id=song_id, n=n)
            return "POPULARITY FALLBACK", list(fallback["song_id"])
        return "HYBRID", list(recs.index)

    elif in_cf:
        recs = get_cf_recommendations(song_id, cf_similarity_df, n=n)
        if len(recs) == 0:
            fallback = get_popular_fallback(popularity_df, exclude_song_id=song_id, n=n)
            return "POPULARITY FALLBACK", list(fallback["song_id"])
        return "COLLABORATIVE FILTERING", list(recs.index)

    elif in_cbf:
        recs = get_cbf_recommendations(song_id, cbf_similarity_df, n=n)
        if len(recs) == 0:
            fallback = get_popular_fallback(popularity_df, exclude_song_id=song_id, n=n)
            return "POPULARITY FALLBACK", list(fallback["song_id"])
        return "CONTENT-BASED FILTERING", list(recs.index)

    else:
        fallback = get_popular_fallback(popularity_df, exclude_song_id=song_id, n=n)
        return "POPULARITY FALLBACK", list(fallback["song_id"])


def initialise_recommender():
    playlist_data, audio_data = load_datasets()
    shared_songs, playlist_only, audio_only = find_overlap(playlist_data, audio_data)

    cf_similarity_df = build_cf_model(playlist_data)
    cbf_similarity_df = build_cbf_model(audio_data)
    popularity_df = build_popularity_table(playlist_data)

    return {
        "playlist_data": playlist_data,
        "audio_data": audio_data,
        "shared_songs": shared_songs,
        "playlist_only": playlist_only,
        "audio_only": audio_only,
        "cf_similarity_df": cf_similarity_df,
        "cbf_similarity_df": cbf_similarity_df,
        "popularity_df": popularity_df
    }