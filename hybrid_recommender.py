import pandas as pd
import re

# CONFIG
PLAYLIST_PATH = "data/spotify_dataset.csv"
AUDIO_PATH = "data/dataset.csv"

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


def build_popularity_table(playlist_data):
    popularity = (
        playlist_data["song_id"]
        .value_counts()
        .reset_index()
    )
    popularity.columns = ["song_id", "count"]
    return popularity


def initialise_recommender():
    playlist_data, audio_data = load_datasets()
    shared_songs, playlist_only, audio_only = find_overlap(playlist_data, audio_data)
    popularity_df = build_popularity_table(playlist_data)

    return {
        "playlist_data": playlist_data,
        "audio_data": audio_data,
        "shared_songs": shared_songs,
        "playlist_only": playlist_only,
        "audio_only": audio_only,
        "popularity_df": popularity_df
    }