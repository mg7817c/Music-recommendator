import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from hybrid_recommender import (
    initialise_recommender,
    recommend_from_song_id,
    recommend_from_playlist,
)

st.set_page_config(
    page_title="Hybrid Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

NUM_RECOMMENDATIONS = 5

FEATURE_LABELS = {
    "energy":           "Energy",
    "valence":          "Mood",
    "danceability":     "Danceability",
    "acousticness":     "Acoustic Feel",
    "instrumentalness": "Instrumental",
    "liveness":         "Live Feel",
    "speechiness":      "Speechiness",
}

MODE_LABELS = {
    "HYBRID":                  "Based on listening patterns and audio similarity",
    "COLLABORATIVE FILTERING": "Based on what similar listeners enjoy",
    "CONTENT-BASED FILTERING": "Based on the sound of your selected song",
    "POPULARITY FALLBACK":     "Popular songs you might enjoy",
}

# dark theme colours
BG      = "#0f0f13"
CARD    = "#1a1a24"
BORDER  = "#2d2d3d"
PURPLE  = "#a78bfa"
TEXT    = "#e5e7eb"
MUTED   = "#6b7280"
WHITE   = "#ffffff"
CARD_HI = "#1e1a3a"
BORD_HI = "#3730a3"

@st.cache_resource(show_spinner=False)
def load_recommender():
    with st.status("Starting up...", expanded=True) as status:
        st.write("Loading datasets...")
        result = initialise_recommender()
        st.write("Building models...")
        status.update(label="Ready!", state="complete", expanded=False)
    return result

state   = load_recommender()
catalog = state["catalog"]

defaults = {
    "selected_song_id":        None,
    "selected_song_display":   None,
    "selected_song_mode":      "",
    "recommendation_mode":     "",
    "recommendations":         [],
    "excluded_songs":          set(),
    "_last_selected":          None,
    "refresh_offset":          0,
    "playlist":                [],
    "playlist_recs":           [],
    "playlist_mode":           "",
    "playlist_excluded":       set(),
    "playlist_refresh_offset": 0,
    "_last_pl_selected":       None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.markdown(f"""
<style>
.stApp {{ background-color: {BG}; }}
[data-testid="stSidebar"] {{ background-color: #13131a; border-right: 1px solid {BORDER}; }}
html, body, [class*="css"] {{ color: {TEXT}; }}
h1, h2, h3 {{ color: {WHITE} !important; }}
[data-testid="stSelectbox"] > div > div {{
    background-color: {CARD} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT} !important;
    border-radius: 10px !important;
}}
</style>
""", unsafe_allow_html=True)


def get_audio_features(song_id):
    audio = state["audio_data"]
    rows  = audio[audio["song_id"] == song_id]
    if rows.empty:
        return None
    available = [c for c in FEATURE_LABELS if c in audio.columns]
    vals = rows[available].iloc[0].copy()
    if "tempo" in vals.index:
        vals["tempo"] = ((vals["tempo"] - 50) / 150).clip(0, 1)
    return vals


def get_blob_colour(song_id):
    audio = state["audio_data"]
    rows  = audio[audio["song_id"] == song_id]
    if rows.empty:
        return "#3730a3", "#6366f1"
    energy  = float(rows["energy"].iloc[0])  if "energy"  in rows.columns else 0.5
    valence = float(rows["valence"].iloc[0]) if "valence" in rows.columns else 0.5
    warmth  = (energy + valence) / 2
    if warmth > 0.7:
        return "#7c3aed", "#f59e0b"
    elif warmth > 0.5:
        return "#6366f1", "#a78bfa"
    elif warmth > 0.3:
        return "#0f6e56", "#1d9e75"
    else:
        return "#0c447c", "#185fa5"


def h(text):
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;"))


def recommendation_card(i, song_id, seed_id, mode, prefix="single"):
    row     = catalog[catalog["song_id"] == song_id]
    display = row["display"].values[0] if not row.empty else song_id.title()
    parts   = display.split(" - ", 1)
    artist  = h(parts[0]) if len(parts) == 2 else ""
    track   = h(parts[1]) if len(parts) == 2 else h(display)

    col1, _   = get_blob_colour(song_id)
    rec_feats = get_audio_features(song_id)

    feat_bars_html = ""
    if rec_feats is not None:
        show_feats   = ["energy", "valence", "danceability", "acousticness"]
        feat_colours = {
            "energy": "#7c3aed", "valence": "#6366f1",
            "danceability": "#8b5cf6", "acousticness": "#a78bfa",
        }
        for feat in show_feats:
            if feat in rec_feats.index:
                pct    = round(float(rec_feats[feat]) * 100)
                colour = feat_colours.get(feat, "#a78bfa")
                label  = FEATURE_LABELS.get(feat, feat)
                feat_bars_html += (
                    f'<div style="margin-bottom:6px;">'
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
                    f'<span style="font-size:10px;color:{MUTED};text-transform:uppercase;">{label}</span>'
                    f'<span style="font-size:10px;color:{MUTED};">{pct}%</span>'
                    f'</div>'
                    f'<div style="height:4px;background:{BORDER};border-radius:2px;">'
                    f'<div style="height:100%;width:{pct}%;background:{colour};border-radius:2px;"></div>'
                    f'</div></div>'
                )

    card_html = (
        f'<div style="background:{CARD};border:1px solid {BORDER};'
        f'border-radius:14px;padding:16px 18px;margin-bottom:8px;">'
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">'
        f'<div style="width:46px;height:46px;border-radius:10px;flex-shrink:0;'
        f'background:{col1};display:flex;align-items:center;justify-content:center;font-size:20px;">&#127925;</div>'
        f'<div style="flex:1;min-width:0;">'
        f'<div style="font-size:15px;font-weight:600;color:{WHITE};">{track}</div>'
        f'<div style="font-size:12px;color:{MUTED};margin-top:2px;">{artist}</div>'
        f'</div>'
        f'<span style="font-size:12px;color:{MUTED};">#{i+1}</span>'
        f'</div>'
        f'{feat_bars_html}'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

    col_exp, col_dismiss = st.columns([5, 1])
    with col_exp:
        with st.expander("Why was this recommended?"):
            st.markdown(f"Mode: **{mode}**")
    with col_dismiss:
        if st.button("✕ Hide", key=f"{prefix}_dismiss_{song_id}_{i}"):
            if prefix == "single":
                st.session_state.excluded_songs.add(song_id)
                st.session_state.recommendations = [
                    r for r in st.session_state.recommendations if r != song_id
                ]
            else:
                st.session_state.playlist_excluded.add(song_id)
                st.session_state.playlist_recs = [
                    r for r in st.session_state.playlist_recs if r != song_id
                ]
            st.rerun()


def run_recommendations(song_id, offset=0):
    fetch_n = NUM_RECOMMENDATIONS + len(st.session_state.excluded_songs) + offset
    mode, recs = recommend_from_song_id(
        song_id=song_id,
        shared_songs=state["shared_songs"],
        playlist_only=state["playlist_only"],
        audio_only=state["audio_only"],
        cf_similarity_df=state["cf_similarity_df"],
        cbf_similarity_df=state["cbf_similarity_df"],
        popularity_df=state["popularity_df"],
        n=fetch_n,
    )
    filtered = [r for r in recs if r not in st.session_state.excluded_songs]
    window   = filtered[offset:offset + NUM_RECOMMENDATIONS]
    if not window:
        st.session_state.refresh_offset = 0
        window = filtered[:NUM_RECOMMENDATIONS]
    return mode, window


def run_playlist_recommendations(offset=0):
    seed_ids = [s["song_id"] for s in st.session_state.playlist]
    fetch_n  = NUM_RECOMMENDATIONS + len(st.session_state.playlist_excluded) + offset
    mode, recs = recommend_from_playlist(
        song_ids=seed_ids,
        cf_similarity_df=state["cf_similarity_df"],
        cbf_similarity_df=state["cbf_similarity_df"],
        popularity_df=state["popularity_df"],
        n=fetch_n,
    )
    filtered = [r for r in recs if r not in st.session_state.playlist_excluded]
    window   = filtered[offset:offset + NUM_RECOMMENDATIONS]
    if not window:
        st.session_state.playlist_refresh_offset = 0
        window = filtered[:NUM_RECOMMENDATIONS]
    return mode, window


st.title("🎵 Hybrid Music Recommender")
st.caption(f"{len(catalog):,} songs available")
st.markdown("---")

tab_single, tab_playlist = st.tabs(["🎵 Single Song", "📋 Playlist Mode"])

with tab_single:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("Search")
        selected_option = st.selectbox(
            "Search", options=catalog["display"].tolist(), index=None,
            placeholder="Type an artist or song title...",
            label_visibility="collapsed", key="single_search",
        )

        if selected_option and selected_option != st.session_state._last_selected:
            st.session_state._last_selected     = selected_option
            row = catalog[catalog["display"] == selected_option].iloc[0]
            st.session_state.selected_song_id      = row["song_id"]
            st.session_state.selected_song_display = row["display"]
            st.session_state.recommendation_mode   = ""
            st.session_state.recommendations       = []
            st.session_state.refresh_offset        = 0
            with st.spinner("Finding recommendations..."):
                mode, recs = run_recommendations(row["song_id"])
            st.session_state.recommendation_mode = mode
            st.session_state.recommendations     = recs

        if st.session_state.selected_song_display:
            st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
            st.subheader("Now playing")
            st.markdown(
                f'<div style="padding:20px;border-radius:12px;background:{CARD_HI};'
                f'border:1px solid {BORD_HI};margin-bottom:8px;">'
                f'<div style="font-size:36px;">🎵</div>'
                f'<div style="font-size:20px;font-weight:bold;color:{WHITE};margin-top:6px;">'
                f'{st.session_state.selected_song_display}</div></div>',
                unsafe_allow_html=True,
            )
            if st.session_state.recommendation_mode:
                st.caption(f"✦ {MODE_LABELS.get(st.session_state.recommendation_mode, '')}")

    with col_right:
        st.subheader("Recommendations")
        if st.session_state.recommendations:
            _, col_refresh = st.columns([3, 1])
            with col_refresh:
                if st.button("🔄 Refresh", use_container_width=True, key="single_refresh"):
                    st.session_state.refresh_offset += NUM_RECOMMENDATIONS
                    with st.spinner("Refreshing..."):
                        mode, recs = run_recommendations(
                            st.session_state.selected_song_id,
                            offset=st.session_state.refresh_offset,
                        )
                    st.session_state.recommendation_mode = mode
                    st.session_state.recommendations     = recs
                    st.rerun()
            for i, song_id in enumerate(st.session_state.recommendations):
                recommendation_card(i, song_id, st.session_state.selected_song_id,
                                    st.session_state.recommendation_mode, "single")
        elif st.session_state.recommendation_mode:
            st.info("No recommendations found.")
        else:
            st.caption("Recommendations appear here once you select a song.")

with tab_playlist:
    col_pl_left, col_pl_right = st.columns([1, 1], gap="large")

    with col_pl_left:
        st.subheader("Build your playlist")
        pl_option = st.selectbox(
            "Add a song", options=catalog["display"].tolist(), index=None,
            placeholder="Search for a song to add...",
            label_visibility="collapsed", key="playlist_search",
        )
        col_add, col_clear = st.columns([1, 1])
        with col_add:
            if st.button("+ Add to playlist", use_container_width=True, key="pl_add"):
                if pl_option is None:
                    st.warning("Select a song first.")
                elif len(st.session_state.playlist) >= 10:
                    st.warning("Playlist limit is 10 songs.")
                else:
                    row      = catalog[catalog["display"] == pl_option].iloc[0]
                    existing = [s["song_id"] for s in st.session_state.playlist]
                    if row["song_id"] in existing:
                        st.warning("Already in playlist.")
                    else:
                        st.session_state.playlist.append({
                            "song_id": row["song_id"],
                            "display": row["display"],
                        })
                        st.session_state.playlist_recs = []
                        st.rerun()
        with col_clear:
            if st.button("Clear playlist", use_container_width=True, key="pl_clear"):
                st.session_state.playlist      = []
                st.session_state.playlist_recs = []
                st.rerun()

        if st.session_state.playlist:
            for idx, song in enumerate(st.session_state.playlist):
                col_n, col_rm = st.columns([5, 1])
                with col_n:
                    st.markdown(f"{idx+1}. {song['display']}")
                with col_rm:
                    if st.button("✕", key=f"pl_remove_{idx}_{song['song_id']}"):
                        st.session_state.playlist.pop(idx)
                        st.session_state.playlist_recs = []
                        st.rerun()

            if st.button("🎵 Recommend from playlist", use_container_width=True, key="pl_recommend"):
                with st.spinner("Finding recommendations..."):
                    mode, recs = run_playlist_recommendations()
                st.session_state.playlist_mode = mode
                st.session_state.playlist_recs = recs
                st.rerun()

    with col_pl_right:
        st.subheader("Recommendations")
        if st.session_state.playlist_recs:
            seed_id = st.session_state.playlist[0]["song_id"] if st.session_state.playlist else None
            for i, song_id in enumerate(st.session_state.playlist_recs):
                recommendation_card(i, song_id, seed_id,
                                    st.session_state.playlist_mode, "playlist")
        elif st.session_state.playlist:
            st.caption("Click 'Recommend from playlist' to get started.")
        else:
            st.caption("Add songs to your playlist first.")

with st.sidebar:
    st.header("Hidden Songs")
    all_excluded = st.session_state.excluded_songs | st.session_state.playlist_excluded
    if not all_excluded:
        st.caption("No songs hidden yet.")
    else:
        for song_id in list(all_excluded):
            row     = catalog[catalog["song_id"] == song_id]
            display = row["display"].values[0] if not row.empty else song_id.title()
            col_n, col_r = st.columns([4, 1])
            with col_n:
                st.markdown(f"<span style='font-size:13px;'>{display}</span>",
                            unsafe_allow_html=True)
            with col_r:
                if st.button("↩", key=f"restore_{song_id}"):
                    st.session_state.excluded_songs.discard(song_id)
                    st.session_state.playlist_excluded.discard(song_id)
                    st.rerun()
        st.markdown("---")
        if st.button("Unhide all", use_container_width=True):
            st.session_state.excluded_songs    = set()
            st.session_state.playlist_excluded = set()
            st.rerun()