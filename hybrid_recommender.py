import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# CONFIG
PLAYLIST_PATH = "data/spotify_dataset.csv"
AUDIO_PATH = "data/dataset.csv"

# increased from 5000 - was causing most test seeds to fall outside the model
CF_MAX_SONGS = 25000
CBF_MAX_SONGS = 25000

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

    # remove text in brackets
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\[.*?\]", "", text)

    # remove feat/ft text
    text = re.sub(r"\bfeat\b.*", "", text)
    text = re.sub(r"\bft\b.*", "", text)
    text = re.sub(r"\bfeaturing\b.*", "", text)

    noisy_words = [
        "remastered", "remaster", "live", "edit", "radio edit",
        "version", "mono", "stereo", "deluxe", "explicit",
        "clean", "bonus track"
    ]
    for word in noisy_words:
        text = text.replace(word, "")

    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_datasets():
    playlist_data = pd.read_csv(PLAYLIST_PATH, engine="python", on_bad_lines="skip")
    audio_data    = pd.read_csv(AUDIO_PATH,    engine="python", on_bad_lines="skip")

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

    required_playlist_cols = ["user", "artist", "track"]
    required_audio_cols = ["artist", "track"]

    for col in required_playlist_cols:
        if col not in playlist_data.columns:
            raise KeyError(f"Missing required column in playlist dataset: {col}")
    for col in required_audio_cols:
        if col not in audio_data.columns:
            raise KeyError(f"Missing required column in audio dataset: {col}")

    playlist_data["artist_clean"] = playlist_data["artist"].apply(clean_text)
    playlist_data["track_clean"]  = playlist_data["track"].apply(clean_text)
    audio_data["artist_clean"]    = audio_data["artist"].apply(clean_text)
    audio_data["track_clean"]     = audio_data["track"].apply(clean_text)

    playlist_data = playlist_data[
        (playlist_data["artist_clean"] != "") &
        (playlist_data["track_clean"]  != "")
    ].copy()
    audio_data = audio_data[
        (audio_data["artist_clean"] != "") &
        (audio_data["track_clean"]  != "")
    ].copy()

    playlist_data["song_id"] = (
        playlist_data["artist_clean"] + " - " + playlist_data["track_clean"]
    )
    audio_data["song_id"] = (
        audio_data["artist_clean"] + " - " + audio_data["track_clean"]
    )

    log("Playlist rows:", len(playlist_data))
    log("Audio rows:",    len(audio_data))

    return playlist_data, audio_data


def find_overlap(playlist_data, audio_data):
    playlist_songs = set(playlist_data["song_id"].unique())
    audio_songs    = set(audio_data["song_id"].unique())

    shared_songs  = playlist_songs.intersection(audio_songs)
    playlist_only = playlist_songs - shared_songs
    audio_only    = audio_songs    - shared_songs

    log("Unique playlist songs:", len(playlist_songs))
    log("Unique audio songs:",    len(audio_songs))
    log("Shared songs:",          len(shared_songs))

    return shared_songs, playlist_only, audio_only


def build_cf_model(playlist_data):
    # keep most common songs for speed
    top_songs      = playlist_data["song_id"].value_counts().head(CF_MAX_SONGS).index
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
    feature_cols = [
        "acousticness", "danceability", "energy", "instrumentalness",
        "liveness", "loudness", "speechiness", "tempo", "valence"
    ]

    missing_cols = [col for col in feature_cols if col not in audio_data.columns]
    if missing_cols:
        raise KeyError(f"Missing audio feature columns in audio dataset: {missing_cols}")

    # some songs appear in multiple rows with different genres
    # averaging gives a more representative feature vector than just keeping one row
    song_features = (
        audio_data
        .groupby("song_id")[feature_cols]
        .mean()
        .reset_index()
    )

    if "popularity" in audio_data.columns:
        popularity_mean = (
            audio_data
            .groupby("song_id")["popularity"]
            .mean()
            .reset_index()
            .rename(columns={"popularity": "popularity_mean"})
        )
        song_features = song_features.merge(popularity_mean, on="song_id", how="left")
        song_features = song_features.sort_values("popularity_mean", ascending=False)

    song_features = song_features.head(CBF_MAX_SONGS).copy()

    X             = song_features[feature_cols].dropna().copy()
    song_features = song_features.loc[X.index].copy()

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cbf_similarity    = cosine_similarity(X_scaled)
    cbf_similarity_df = pd.DataFrame(
        cbf_similarity,
        index=song_features["song_id"],
        columns=song_features["song_id"]
    )

    return cbf_similarity_df


def build_popularity_table(playlist_data, audio_data=None):
    popularity = (
        playlist_data["song_id"]
        .value_counts()
        .reset_index()
    )
    popularity.columns = ["song_id", "count"]

    # combine playlist frequency with Spotify popularity score
    # so songs only in the audio dataset can still appear in the fallback
    if audio_data is not None and "popularity" in audio_data.columns:
        audio_pop = (
            audio_data
            .groupby("song_id")["popularity"]
            .mean()
            .reset_index()
            .rename(columns={"popularity": "audio_popularity"})
        )
        popularity = popularity.merge(audio_pop, on="song_id", how="outer")
        popularity["count"]            = popularity["count"].fillna(0)
        popularity["audio_popularity"] = popularity["audio_popularity"].fillna(0)

        max_count = popularity["count"].max()            or 1
        max_audio = popularity["audio_popularity"].max() or 1

        popularity["score"] = (
            0.5 * popularity["count"]            / max_count +
            0.5 * popularity["audio_popularity"] / max_audio
        )
        popularity = popularity.sort_values("score", ascending=False).reset_index(drop=True)
    else:
        popularity = popularity.sort_values("count", ascending=False).reset_index(drop=True)

    return popularity


def build_song_catalog(cf_similarity_df, cbf_similarity_df):
    cf_index  = set(cf_similarity_df.index)
    cbf_index = set(cbf_similarity_df.index)

    all_song_ids = sorted(cf_index.union(cbf_index))

    rows = []
    for song_id in all_song_ids:
        in_cf  = song_id in cf_index
        in_cbf = song_id in cbf_index

        if in_cf and in_cbf:
            mode = "HYBRID"
        elif in_cf:
            mode = "CF"
        else:
            mode = "CBF"

        parts = song_id.split(" - ", 1)
        if len(parts) == 2:
            artist_display, track_display = parts
        else:
            artist_display, track_display = song_id, ""

        display = f"{artist_display.title()} - {track_display.title()}"

        rows.append({
            "song_id":        song_id,
            "artist_display": artist_display,
            "track_display":  track_display,
            "display":        display,
            "mode":           mode
        })

    catalog = pd.DataFrame(rows)
    return catalog.sort_values("display").reset_index(drop=True)


def search_songs(catalog, query, limit=50):
    query = clean_text(query)
    if not query:
        return catalog.head(limit)
    mask = (
        catalog["song_id"].str.contains(query, case=False, na=False) |
        catalog["display"].str.contains(query, case=False, na=False)
    )
    return catalog[mask].head(limit)


def _normalise(scores):
    # min-max normalise to [0,1] before combining scores
    # without this, whichever model has higher raw values dominates
    min_val = scores.min()
    max_val = scores.max()
    if max_val - min_val < 1e-9:
        return scores * 0.0
    return (scores - min_val) / (max_val - min_val)


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

    cf_scores  = cf_similarity_df[song_id]
    cbf_scores = cbf_similarity_df[song_id]

    common_index = cf_scores.index.intersection(cbf_scores.index)

    if len(common_index) < n:
        log("Common index too small, falling back to CF")
        return pd.Series(dtype=float)

    cf_norm  = _normalise(cf_scores.loc[common_index])
    cbf_norm = _normalise(cbf_scores.loc[common_index])

    hybrid_scores = ALPHA_CF * cf_norm + ALPHA_CBF * cbf_norm

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
    in_cf  = song_id in cf_similarity_df.index
    in_cbf = song_id in cbf_similarity_df.index

    log(f"\nSelected: {song_id}")
    log(f"In CF: {in_cf} | In CBF: {in_cbf}")

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


def recommend_from_playlist(
    song_ids,
    cf_similarity_df,
    cbf_similarity_df,
    popularity_df,
    n=10,
):
    # averages similarity vectors across all seed songs
    # so recommendations reflect the whole playlist not just one track
    cf_scores_list  = []
    cbf_scores_list = []

    for song_id in song_ids:
        if song_id in cf_similarity_df.index:
            cf_scores_list.append(cf_similarity_df[song_id])
        if song_id in cbf_similarity_df.index:
            cbf_scores_list.append(cbf_similarity_df[song_id])

    if cf_scores_list and cbf_scores_list:
        cf_avg  = pd.concat(cf_scores_list,  axis=1).mean(axis=1)
        cbf_avg = pd.concat(cbf_scores_list, axis=1).mean(axis=1)
        common  = cf_avg.index.union(cbf_avg.index)
        scores  = (
            ALPHA_CF  * cf_avg.reindex(common, fill_value=0) +
            ALPHA_CBF * cbf_avg.reindex(common, fill_value=0)
        )
        mode = "HYBRID"

    elif cf_scores_list:
        scores = pd.concat(cf_scores_list, axis=1).mean(axis=1)
        mode   = "COLLABORATIVE FILTERING"

    elif cbf_scores_list:
        scores = pd.concat(cbf_scores_list, axis=1).mean(axis=1)
        mode   = "CONTENT-BASED FILTERING"

    else:
        fallback = get_popular_fallback(popularity_df, n=n)
        return "POPULARITY FALLBACK", list(fallback["song_id"])

    # remove seed songs from results
    scores = scores.drop(labels=list(song_ids), errors="ignore")
    recs   = list(scores.sort_values(ascending=False).head(n).index)

    if not recs:
        fallback = get_popular_fallback(popularity_df, n=n)
        return "POPULARITY FALLBACK", list(fallback["song_id"])

    return mode, recs


def initialise_recommender():
    playlist_data, audio_data = load_datasets()
    shared_songs, playlist_only, audio_only = find_overlap(playlist_data, audio_data)

    cf_similarity_df  = build_cf_model(playlist_data)
    cbf_similarity_df = build_cbf_model(audio_data)
    popularity_df     = build_popularity_table(playlist_data, audio_data)
    catalog           = build_song_catalog(cf_similarity_df, cbf_similarity_df)

    return {
        "playlist_data":     playlist_data,
        "audio_data":        audio_data,
        "shared_songs":      shared_songs,
        "playlist_only":     playlist_only,
        "audio_only":        audio_only,
        "cf_similarity_df":  cf_similarity_df,
        "cbf_similarity_df": cbf_similarity_df,
        "popularity_df":     popularity_df,
        "catalog":           catalog
    }