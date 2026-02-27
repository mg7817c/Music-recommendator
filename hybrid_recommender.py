import pandas as pd

# CONFIG
PLAYLIST_PATH = "data/spotify_dataset.csv"
AUDIO_PATH = "data/dataset.csv"

TOP_N = 10
VERBOSE = False


def log(*args):
    if VERBOSE:
        print(*args)


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

    return playlist_data, audio_data


def initialise_recommender():
    playlist_data, audio_data = load_datasets()

    return {
        "playlist_data": playlist_data,
        "audio_data": audio_data
    }