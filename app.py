import streamlit as st
import pandas as pd
from hybrid_recommender import (
    initialise_recommender,
    recommend_from_song_id,
)

st.set_page_config(
    page_title="Hybrid Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

NUM_RECOMMENDATIONS = 5

FEATURE_LABELS = {
    "energy":       "Energy",
    "valence":      "Mood",
    "danceability": "Danceability",
    "acousticness": "Acoustic Feel",
}

MODE_LABELS = {
    "HYBRID":                  "Based on listening patterns and audio similarity",
    "COLLABORATIVE FILTERING": "Based on what similar listeners enjoy",
    "CONTENT-BASED FILTERING": "Based on the sound of your selected song",
    "POPULARITY FALLBACK":     "Popular songs you might enjoy",
}

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
    "selected_song_id":      None,
    "selected_song_display": None,
    "recommendation_mode":   "",
    "recommendations":       [],
    "excluded_songs":        set(),
    "_last_selected":        None,
    "refresh_offset":        0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def get_audio_features(song_id):
    audio = state["audio_data"]
    rows  = audio[audio["song_id"] == song_id]
    if rows.empty:
        return None
    available = [c for c in FEATURE_LABELS if c in audio.columns]
    return rows[available].iloc[0].copy()


def h(text):
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))


def recommendation_card(i, song_id, mode):
    row     = catalog[catalog["song_id"] == song_id]
    display = row["display"].values[0] if not row.empty else song_id.title()
    parts   = display.split(" - ", 1)
    artist  = h(parts[0]) if len(parts) == 2 else ""
    track   = h(parts[1]) if len(parts) == 2 else h(display)

    feats     = get_audio_features(song_id)
    feat_html = ""
    if feats is not None:
        for feat, label in FEATURE_LABELS.items():
            if feat in feats.index:
                pct = round(float(feats[feat]) * 100)
                feat_html += (
                    f'<div style="margin-bottom:5px;">'
                    f'<span style="font-size:11px;color:#6b7280;">{label}: {pct}%</span>'
                    f'<div style="height:4px;background:#2d2d3d;border-radius:2px;">'
                    f'<div style="height:100%;width:{pct}%;background:#a78bfa;border-radius:2px;"></div>'
                    f'</div></div>'
                )

    card_html = (
        f'<div style="background:#1a1a24;border:1px solid #2d2d3d;'
        f'border-radius:12px;padding:14px 16px;margin-bottom:8px;">'
        f'<div style="font-size:15px;font-weight:600;color:#ffffff;">#{i+1} {track}</div>'
        f'<div style="font-size:12px;color:#6b7280;margin-top:2px;">{artist}</div>'
        f'<div style="margin-top:10px;">{feat_html}</div>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

    if st.button("✕ Hide", key=f"dismiss_{song_id}_{i}"):
        st.session_state.excluded_songs.add(song_id)
        st.session_state.recommendations = [
            r for r in st.session_state.recommendations if r != song_id
        ]
        st.rerun()


st.title("🎵 Hybrid Music Recommender")
st.caption(f"{len(catalog):,} songs available")
st.markdown("---")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Search")
    selected_option = st.selectbox(
        "Search", options=catalog["display"].tolist(), index=None,
        placeholder="Type an artist or song title...",
        label_visibility="collapsed", key="single_search",
    )

    if selected_option and selected_option != st.session_state._last_selected:
        st.session_state._last_selected    = selected_option
        row = catalog[catalog["display"] == selected_option].iloc[0]
        st.session_state.selected_song_id      = row["song_id"]
        st.session_state.selected_song_display = row["display"]
        st.session_state.recommendation_mode   = ""
        st.session_state.recommendations       = []
        st.session_state.refresh_offset        = 0

        with st.spinner("Finding recommendations..."):
            mode, recs = recommend_from_song_id(
                song_id=row["song_id"],
                shared_songs=state["shared_songs"],
                playlist_only=state["playlist_only"],
                audio_only=state["audio_only"],
                cf_similarity_df=state["cf_similarity_df"],
                cbf_similarity_df=state["cbf_similarity_df"],
                popularity_df=state["popularity_df"],
                n=NUM_RECOMMENDATIONS + len(st.session_state.excluded_songs),
            )
        st.session_state.recommendation_mode = mode
        st.session_state.recommendations = [
            r for r in recs if r not in st.session_state.excluded_songs
        ][:NUM_RECOMMENDATIONS]

    if st.session_state.selected_song_display:
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        st.subheader("Now playing")
        st.info(st.session_state.selected_song_display)
        if st.session_state.recommendation_mode:
            st.caption(f"✦ {MODE_LABELS.get(st.session_state.recommendation_mode, '')}")

with col_right:
    st.subheader("Recommendations")

    if st.session_state.recommendations:
        for i, song_id in enumerate(st.session_state.recommendations):
            recommendation_card(i, song_id, st.session_state.recommendation_mode)
    elif st.session_state.recommendation_mode:
        st.info("No recommendations found.")
    else:
        st.caption("Recommendations appear here once you select a song.")

with st.sidebar:
    st.header("Hidden Songs")
    all_excluded = st.session_state.excluded_songs
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
                    st.rerun()