import streamlit as st
from hybrid_recommender import (
    initialise_recommender,
    recommend_from_song_id,
)

st.set_page_config(
    page_title="Hybrid Music Recommender",
    page_icon="🎵",
    layout="wide",
)

NUM_RECOMMENDATIONS = 5

MODE_LABELS = {
    "HYBRID":                  "Based on listening patterns and audio similarity",
    "COLLABORATIVE FILTERING": "Based on what similar listeners enjoy",
    "CONTENT-BASED FILTERING": "Based on the sound of your selected song",
    "POPULARITY FALLBACK":     "Popular songs you might enjoy",
}

@st.cache_resource(show_spinner=False)
def load_recommender():
    with st.spinner("Loading recommender system..."):
        return initialise_recommender()

state   = load_recommender()
catalog = state["catalog"]

if "selected_song_id" not in st.session_state:
    st.session_state.selected_song_id      = None
if "selected_song_display" not in st.session_state:
    st.session_state.selected_song_display = None
if "recommendation_mode" not in st.session_state:
    st.session_state.recommendation_mode  = ""
if "recommendations" not in st.session_state:
    st.session_state.recommendations      = []
if "_last_selected" not in st.session_state:
    st.session_state._last_selected       = None

st.title("🎵 Hybrid Music Recommender")
st.caption(f"{len(catalog):,} songs available")
st.markdown("---")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Search for a song")

    selected_option = st.selectbox(
        "Search",
        options=catalog["display"].tolist(),
        index=None,
        placeholder="Type an artist or song title...",
        label_visibility="collapsed",
        key="single_search",
    )

    if selected_option and selected_option != st.session_state._last_selected:
        st.session_state._last_selected = selected_option
        row = catalog[catalog["display"] == selected_option].iloc[0]
        st.session_state.selected_song_id      = row["song_id"]
        st.session_state.selected_song_display = row["display"]
        st.session_state.recommendation_mode   = ""
        st.session_state.recommendations       = []

        with st.spinner("Finding recommendations..."):
            mode, recs = recommend_from_song_id(
                song_id=row["song_id"],
                shared_songs=state["shared_songs"],
                playlist_only=state["playlist_only"],
                audio_only=state["audio_only"],
                cf_similarity_df=state["cf_similarity_df"],
                cbf_similarity_df=state["cbf_similarity_df"],
                popularity_df=state["popularity_df"],
                n=NUM_RECOMMENDATIONS,
            )
        st.session_state.recommendation_mode = mode
        st.session_state.recommendations     = recs

    if st.session_state.selected_song_display:
        st.markdown("**Now playing:**")
        st.info(st.session_state.selected_song_display)
        if st.session_state.recommendation_mode:
            st.caption(
                f"✦ {MODE_LABELS.get(st.session_state.recommendation_mode, '')}"
            )

with col_right:
    st.subheader("Recommendations")

    if st.session_state.recommendations:
        for i, song_id in enumerate(st.session_state.recommendations):
            row     = catalog[catalog["song_id"] == song_id]
            display = row["display"].values[0] if not row.empty else song_id.title()
            mode    = row["mode"].values[0]    if not row.empty else ""
            st.markdown(f"**{i+1}.** {display}  `{mode}`")
    elif st.session_state.recommendation_mode:
        st.info("No recommendations found.")
    else:
        st.caption("Recommendations appear here once you select a song.")