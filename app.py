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

DARK = {
    "bg":        "#0f0f13",
    "sidebar":   "#13131a",
    "card":      "#1a1a24",
    "card_now":  "#1e1a3a",
    "border":    "#2d2d3d",
    "border_hi": "#3730a3",
    "purple":    "#a78bfa",
    "purple_dk": "#7c3aed",
    "text":      "#e5e7eb",
    "muted":     "#6b7280",
    "white":     "#ffffff",
}

LIGHT = {
    "bg":        "#f5f3ff",
    "sidebar":   "#ede9fe",
    "card":      "#ffffff",
    "card_now":  "#ede9fe",
    "border":    "#c4b5fd",
    "border_hi": "#7c3aed",
    "purple":    "#5b21b6",
    "purple_dk": "#4c1d95",
    "text":      "#1e1b4b",
    "muted":     "#4b5563",
    "white":     "#1e1b4b",
}

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


# load everything once on startup

@st.cache_resource(show_spinner=False)
def load_recommender():
    with st.status("Starting up...", expanded=True) as status:
        st.write("Loading datasets...")
        result = initialise_recommender()
        st.write("Building recommendation models...")
        status.update(label="Ready!", state="complete", expanded=False)
    return result


state   = load_recommender()
catalog = state["catalog"]

# session state defaults - must come before T assignment
# Must come before T assignment so dark_mode is available

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
    "dark_mode":               True,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# pick theme based on toggle
T = DARK if st.session_state.dark_mode else LIGHT

# inject theme CSS
st.markdown(f"""
<style>
.stApp {{ background-color: {T['bg']}; }}
[data-testid="stSidebar"] {{
    background-color: {T['sidebar']};
    border-right: 1px solid {T['border']};
}}
html, body, [class*="css"] {{ color: {T['text']}; }}
h1 {{ color: {T['white']} !important; letter-spacing: -0.5px; }}
h2, h3 {{ color: {T['white']} !important; }}
[data-testid="stCaptionContainer"] p {{ color: {T['purple']} !important; }}
[data-testid="stSelectbox"] > div > div {{
    background-color: {T['card']} !important;
    border: 1px solid {T['border']} !important;
    color: {T['text']} !important;
    border-radius: 10px !important;
}}
[data-testid="stBaseButton-secondary"] {{
    background-color: {T['card_now']} !important;
    border: 1px solid {T['border_hi']} !important;
    color: {T['purple']} !important;
    border-radius: 8px !important;
}}
[data-testid="stBaseButton-secondary"]:hover {{
    background-color: {T['card_now']} !important;
    border-color: {T['purple_dk']} !important;
}}
[data-testid="stExpander"] {{
    background-color: {T['card']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 8px !important;
}}
[data-testid="stExpander"] summary {{
    color: {T['purple']} !important;
    background-color: {T['card']} !important;
}}
[data-testid="stExpander"] summary:hover {{
    background-color: {T['card_now']} !important;
}}
[data-testid="stExpander"] details {{
    background-color: {T['card']} !important;
}}
[data-testid="stExpander"] > div,
[data-testid="stExpander"] > div > div,
[data-testid="stExpander"] > div > div > div {{
    background-color: {T['card']} !important;
    color: {T['text']} !important;
}}
[data-testid="stExpander"] p {{ color: {T['text']} !important; }}
[data-testid="stExpander"] span {{ color: {T['text']} !important; }}
[data-testid="stExpander"] div {{ 
    background-color: {T['card']} !important;
    color: {T['text']} !important;
}}
[data-testid="stExpanderDetails"] {{
    background-color: {T['card']} !important;
}}
[data-testid="stExpanderDetails"] > div {{
    background-color: {T['card']} !important;
    color: {T['text']} !important;
}}
[data-testid="stAlert"] {{
    background-color: {T['card_now']} !important;
    border-color: {T['border_hi']} !important;
    color: {T['purple']} !important;
}}
hr {{ border-color: {T['border']} !important; }}
[data-testid="stStatus"] {{
    background-color: {T['card']} !important;
    border-color: {T['border']} !important;
}}
/* Tab text */
[data-testid="stTab"] p, button[data-baseweb="tab"] {{
    color: {T['text']} !important;
}}
/* Selectbox input text as you type */
[data-testid="stSelectbox"] input {{
    color: {T['text']} !important;
    background-color: {T['card']} !important;
}}
/* Selectbox selected value text */
[data-testid="stSelectbox"] div[data-baseweb="select"] span {{
    color: {T['text']} !important;
}}
/* Selectbox dropdown container */
[data-baseweb="popover"] {{
    background-color: {T['card']} !important;
}}
/* Dropdown option text */
[data-baseweb="popover"] li {{
    background-color: {T['card']} !important;
    color: {T['text']} !important;
}}
/* Dropdown option hover */
[data-baseweb="popover"] li:hover {{
    background-color: {T['card_now']} !important;
    color: {T['text']} !important;
}}
/* Dropdown option text spans */
[data-baseweb="popover"] li span {{
    color: {T['text']} !important;
}}
/* Search placeholder text */
[data-testid="stSelectbox"] input::placeholder {{
    color: {T['muted']} !important;
    opacity: 1 !important;
}}
/* General text inputs and labels */
label, p, span, div {{
    color: {T['text']};
}}
/* Sidebar text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {{
    color: {T['text']} !important;
}}
/* Caption text */
small {{ color: {T['muted']} !important; }}
/* Metric labels */
[data-testid="stMetricLabel"] {{ color: {T['text']} !important; }}
[data-testid="stMetricValue"] {{ color: {T['white']} !important; }}
</style>
""", unsafe_allow_html=True)



# helper functions

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


def feature_comparison_chart(seed_id, rec_id):
    # radar chart showing audio profile of seed vs recommended song
    seed_vals = get_audio_features(seed_id)
    rec_vals  = get_audio_features(rec_id)
    if seed_vals is None or rec_vals is None:
        return None

    features = list(seed_vals.index)
    labels   = [FEATURE_LABELS.get(f, f) for f in features]

    seed_values = seed_vals.tolist() + [seed_vals.iloc[0]]
    rec_values  = rec_vals.reindex(features, fill_value=0).tolist() + [rec_vals.reindex(features, fill_value=0).iloc[0]]
    labels_closed = labels + [labels[0]]

    # Use theme colours
    is_dark      = st.session_state.dark_mode
    bg_colour    = T["card"]
    grid_colour  = T["border"]
    tick_colour  = T["muted"]
    legend_colour = T["text"]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=seed_values,
        theta=labels_closed,
        fill="toself",
        name="Your song",
        line=dict(color="#6366f1", width=2),
        fillcolor="rgba(99,102,241,0.15)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=rec_values,
        theta=labels_closed,
        fill="toself",
        name="Recommendation",
        line=dict(color="#10b981", width=2),
        fillcolor="rgba(16,185,129,0.15)",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=bg_colour,
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(color=tick_colour, size=9),
                gridcolor=grid_colour,
                linecolor=grid_colour,
            ),
            angularaxis=dict(
                tickfont=dict(color=tick_colour, size=11),
                gridcolor=grid_colour,
                linecolor=grid_colour,
            ),
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.05,
            xanchor="right", x=1,
            font=dict(color=legend_colour, size=11),
        ),
        height=300,
        margin=dict(l=40, r=40, t=40, b=20),
        paper_bgcolor=bg_colour,
        plot_bgcolor=bg_colour,
        font=dict(color=legend_colour),
    )
    return fig


def build_explanation(seed_id, rec_id, mode):
    cf_sim_df  = state["cf_similarity_df"]
    audio_data = state["audio_data"]

    if mode == "POPULARITY FALLBACK":
        return (
            "This song doesn't have enough listening data or audio information "
            "to match your selection directly. It appears here as a popular "
            "track other listeners enjoy."
        )

    parts = []

    if mode in ("HYBRID", "COLLABORATIVE FILTERING"):
        if seed_id in cf_sim_df.index and rec_id in cf_sim_df.index:
            sim      = round(float(cf_sim_df.loc[seed_id, rec_id]), 3)
            strength = ("very strong" if sim > 0.6
                        else "moderate" if sim > 0.3 else "some")
            parts.append(
                f"Shared listening patterns ({strength} match, "
                f"similarity score: {sim}) — listeners who play your "
                f"selected song also play this one."
            )

    if mode in ("HYBRID", "CONTENT-BASED FILTERING"):
        friendly = {
            "energy": "energy", "valence": "mood (valence)", "tempo": "tempo",
            "danceability": "danceability", "acousticness": "acoustic feel",
            "instrumentalness": "instrumental character",
            "liveness": "live atmosphere", "loudness": "loudness",
            "speechiness": "speechiness",
        }
        pct_features = {
            "energy", "valence", "danceability", "acousticness",
            "instrumentalness", "liveness", "speechiness",
        }
        seed_rows = audio_data[audio_data["song_id"] == seed_id]
        rec_rows  = audio_data[audio_data["song_id"] == rec_id]

        if not seed_rows.empty and not rec_rows.empty:
            available = [c for c in friendly if c in audio_data.columns]
            sv        = seed_rows[available].iloc[0]
            rv        = rec_rows[available].iloc[0]
            top_feats = ((sv - rv).abs() / sv.abs().replace(0, 1))\
                        .nsmallest(3).index.tolist()
            feat_strs = []
            for feat in top_feats:
                s, r  = sv[feat], rv[feat]
                label = friendly[feat]
                if feat in pct_features:
                    feat_strs.append(f"{label} ({s:.0%} vs {r:.0%})")
                elif feat == "tempo":
                    feat_strs.append(f"{label} ({s:.0f} vs {r:.0f} BPM)")
                elif feat == "loudness":
                    feat_strs.append(f"{label} ({s:.1f} vs {r:.1f} dB)")
                else:
                    feat_strs.append(f"{label} ({s:.2f} vs {r:.2f})")
            if feat_strs:
                parts.append("Similar audio features — " +
                             ", ".join(feat_strs) + ".")

    return ("Recommended because of: " + " | ".join(parts)) if parts \
        else "Recommended based on your selection."


def h(text):
    # escape text before injecting into HTML
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;"))


def get_blob_colour(song_id):
    # colour is derived from energy + valence so it actually means something
    # warm colours = high energy/happy, cool colours = low energy/sad
    audio = state["audio_data"]
    rows  = audio[audio["song_id"] == song_id]
    if rows.empty:
        return "#3730a3", "#6366f1"

    energy  = float(rows["energy"].iloc[0])  if "energy"  in rows.columns else 0.5
    valence = float(rows["valence"].iloc[0]) if "valence" in rows.columns else 0.5

    warmth = (energy + valence) / 2

    if warmth > 0.7:
        return "#7c3aed", "#f59e0b"
    elif warmth > 0.5:
        return "#6366f1", "#a78bfa"
    elif warmth > 0.3:
        return "#0f6e56", "#1d9e75"
    else:
        return "#0c447c", "#185fa5"


def get_similarity_pct(seed_id, rec_id, mode):
    # returns a 0-100 similarity score for the circular meter on each card
    cf_sim_df  = state["cf_similarity_df"]
    cbf_sim_df = state["cbf_similarity_df"]

    if (mode in ("HYBRID", "COLLABORATIVE FILTERING")
            and seed_id in cf_sim_df.index
            and rec_id in cf_sim_df.index):
        sim = float(cf_sim_df.loc[seed_id, rec_id])
        return round(sim * 100)

    if (mode in ("HYBRID", "CONTENT-BASED FILTERING")
            and seed_id in cbf_sim_df.index
            and rec_id in cbf_sim_df.index):
        sim = float(cbf_sim_df.loc[seed_id, rec_id])
        return round(sim * 100)

    return None


def get_energy_label(song_id):
    # text label for screen readers since the blob colour alone isn't accessible
    audio = state["audio_data"]
    rows  = audio[audio["song_id"] == song_id]
    if rows.empty:
        return ""
    energy  = float(rows["energy"].iloc[0])  if "energy"  in rows.columns else 0.5
    valence = float(rows["valence"].iloc[0]) if "valence" in rows.columns else 0.5
    warmth  = (energy + valence) / 2
    if warmth > 0.7:
        return "High energy"
    elif warmth > 0.5:
        return "Mid energy"
    elif warmth > 0.3:
        return "Mellow"
    else:
        return "Low energy"


def recommendation_card(i, song_id, seed_id, mode, prefix="single", total=5):
    row     = catalog[catalog["song_id"] == song_id]
    display = row["display"].values[0] if not row.empty else song_id.title()

    parts        = display.split(" - ", 1)
    artist_raw   = parts[0] if len(parts) == 2 else ""
    track_raw    = parts[1] if len(parts) == 2 else display
    artist       = h(artist_raw)
    track        = h(track_raw)

    col1, col2   = get_blob_colour(song_id)
    energy_label = get_energy_label(song_id)
    sim_pct      = get_similarity_pct(seed_id, song_id, mode)
    rec_feats    = get_audio_features(song_id)

    # Spotify search link
    spotify_query   = (artist_raw + " " + track_raw).replace(" ", "%20")
    spotify_url     = f"https://open.spotify.com/search/{spotify_query}"

    # Mode badge — shows which model recommended this song (accessibility + transparency)
    mode_colours = {
        "HYBRID":                  ("#a78bfa", "Hybrid"),
        "COLLABORATIVE FILTERING": ("#6366f1", "CF"),
        "CONTENT-BASED FILTERING": ("#10b981", "CBF"),
        "POPULARITY FALLBACK":     ("#6b7280", "Popular"),
    }
    mode_col, mode_short = mode_colours.get(mode, ("#6b7280", "?"))
    mode_badge = (
        f'<span style="font-size:9px;font-weight:600;color:{mode_col};'
        f'background:{mode_col}22;border-radius:6px;padding:2px 6px;'
        f'letter-spacing:.04em;" title="Recommended by {mode}">{mode_short}</span>'
    )

    # Rank strength label
    rank_colours = {
        0: ("#10b981", "Best match"),
        1: ("#6366f1", "Strong match"),
        2: ("#8b5cf6", "Good match"),
        3: ("#a78bfa", "Fair match"),
        4: ("#6b7280", "Related"),
    }
    rank_col, rank_label = rank_colours.get(i, ("#6b7280", "Related"))
    rank_badge = (
        f'<span style="font-size:10px;font-weight:500;color:{rank_col};'
        f'background:{rank_col}22;border-radius:8px;padding:2px 8px;'
        f'margin-left:4px;">{rank_label}</span>'
    )

    # Feature bars
    bar_bg    = T["border"] if not st.session_state.dark_mode else "#2d2d3d"
    bar_text  = T["muted"]
    bar_pct   = T["muted"]

    feat_bars_html = ""
    if rec_feats is not None:
        show_feats   = ["energy", "valence", "danceability", "acousticness"]
        feat_colours = {
            "energy":       "#7c3aed",
            "valence":      "#6366f1",
            "danceability": "#8b5cf6",
            "acousticness": "#a78bfa",
        }
        for feat in show_feats:
            if feat in rec_feats.index:
                val    = float(rec_feats[feat])
                pct    = round(val * 100)
                colour = feat_colours.get(feat, "#a78bfa")
                label  = FEATURE_LABELS.get(feat, feat)
                feat_bars_html += (
                    f'<div style="margin-bottom:6px;">'
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
                    f'<span style="font-size:10px;color:{bar_text};text-transform:uppercase;'
                    f'letter-spacing:.05em;">{label}</span>'
                    f'<span style="font-size:10px;color:{bar_pct};">{pct}%</span>'
                    f'</div>'
                    f'<div style="height:4px;background:{bar_bg};border-radius:2px;overflow:hidden;"'
                    f' role="progressbar" aria-valuenow="{pct}" aria-valuemin="0" aria-valuemax="100"'
                    f' aria-label="{label}: {pct}%">'
                    f'<div style="height:100%;width:{pct}%;background:{colour};border-radius:2px;"></div>'
                    f'</div></div>'
                )
    else:
        cf_sim_df = state["cf_similarity_df"]
        if seed_id in cf_sim_df.index and song_id in cf_sim_df.index:
            sim      = float(cf_sim_df.loc[seed_id, song_id])
            sim_pct2 = round(sim * 100)
            strength = ("Very strong" if sim > 0.6 else "Strong" if sim > 0.4
                        else "Moderate" if sim > 0.2 else "Mild")
            feat_bars_html = (
                f'<div style="margin-bottom:6px;">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
                f'<span style="font-size:10px;color:{bar_text};text-transform:uppercase;'
                f'letter-spacing:.05em;">Listening connection</span>'
                f'<span style="font-size:10px;color:{bar_pct};">{strength} ({sim:.2f})</span>'
                f'</div>'
                f'<div style="height:4px;background:{bar_bg};border-radius:2px;overflow:hidden;"'
                f' role="progressbar" aria-valuenow="{sim_pct2}" aria-valuemin="0" aria-valuemax="100"'
                f' aria-label="Listening connection: {strength}">'
                f'<div style="height:100%;width:{sim_pct2}%;background:#6366f1;border-radius:2px;"></div>'
                f'</div>'
                f'<div style="font-size:11px;color:{bar_text};margin-top:5px;">'
                f'Listeners who play your selected song also frequently play this one'
                f'</div>'
                f'</div>'
            )

    # Similarity meter
    sim_html = ""
    if sim_pct is not None:
        circ   = 113
        filled = round(circ * sim_pct / 100)
        empty  = circ - filled
        sim_html = (
            f'<div style="display:inline-flex;align-items:center;gap:6px;'
            f'background:{T["card_now"]};border:1px solid {T["border_hi"]};'
            f'border-radius:20px;padding:4px 10px;font-size:11px;color:{T["purple"]};'
            f'margin-bottom:10px;" role="img" aria-label="{sim_pct}% similarity match">'
            f'<svg width="16" height="16" viewBox="0 0 40 40" aria-hidden="true">'
            f'<circle cx="20" cy="20" r="18" fill="none" stroke="{T["border_hi"]}" stroke-width="4"/>'
            f'<circle cx="20" cy="20" r="18" fill="none" stroke="{T["purple"]}" stroke-width="4"'
            f' stroke-dasharray="{filled} {empty}" stroke-dashoffset="28"'
            f' stroke-linecap="round" transform="rotate(-90 20 20)"/>'
            f'</svg>'
            f'{sim_pct}% match</div>'
        )

    card_html = (
        f'<div style="background:{T["card"]};border:1px solid {T["border"]};'
        f'border-radius:14px;padding:16px 18px;margin-bottom:8px;"'
        f' role="article" aria-label="Recommendation {i+1}: {track_raw} by {artist_raw}">'

        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">'

        # Colour blob — music note only, colour encodes energy/mood
        f'<div style="width:46px;height:46px;border-radius:10px;flex-shrink:0;'
        f'background:{col1};display:flex;align-items:center;'
        f'justify-content:center;" aria-label="{energy_label}">'
        f'<span style="font-size:22px;" aria-hidden="true">&#127925;</span>'
        f'</div>'

        f'<div style="flex:1;min-width:0;">'
        f'<div style="font-size:15px;font-weight:600;color:{T["white"]};'
        f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{track}</div>'
        f'<div style="font-size:12px;color:{T["muted"]};margin-top:2px;">{artist}</div>'
        f'<div style="margin-top:4px;">{mode_badge}</div>'
        f'</div>'

        f'<div style="display:flex;align-items:center;flex-shrink:0;gap:4px;">'
        f'<span style="font-size:12px;color:{T["muted"]};">#{i + 1}</span>'
        f'{rank_badge}'
        f'</div>'
        f'</div>'

        f'{sim_html}'
        f'{feat_bars_html}'

        f'</div>'
    )

    st.markdown(card_html, unsafe_allow_html=True)

    col_exp, col_dismiss = st.columns([5, 1])

    with col_exp:
        with st.expander("Why was this recommended?"):
            explanation = build_explanation(seed_id, song_id, mode)
            st.markdown(
                f'<div style="font-size:13px;color:{T["text"]};line-height:1.6;'
                f'padding:4px 0 8px 0;">{explanation}</div>',
                unsafe_allow_html=True,
            )
            fig = feature_comparison_chart(seed_id, song_id)
            if fig:
                st.markdown(
                    f'<div style="font-size:11px;color:{T["muted"]};'
                    f'margin-bottom:4px;">Full audio profile comparison</div>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})

    with col_dismiss:
        if st.button("✕ Hide", key=f"{prefix}_dismiss_{song_id}_{i}",
                     help="Not interested"):
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


# page header and theme toggle

col_title, col_toggle = st.columns([6, 1])
with col_title:
    st.title("🎵 Hybrid Music Recommender")
    st.caption(f"{len(catalog):,} songs available")
with col_toggle:
    st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)
    icon = "☀️" if st.session_state.dark_mode else "🌙"
    label = f"{icon} {'Light' if st.session_state.dark_mode else 'Dark'}"
    if st.button(label, key="theme_toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

st.markdown("---")

tab_single, tab_playlist = st.tabs(["🎵 Single Song", "📋 Playlist Mode"])


# single song tab

with tab_single:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("Search")
        selected_option = st.selectbox(
            "Search", options=catalog["display"].tolist(), index=None,
            placeholder="Type an artist or song title...",
            label_visibility="collapsed", key="single_search",
        )

        if (selected_option
                and selected_option != st.session_state["_last_selected"]):
            st.session_state["_last_selected"]      = selected_option
            row = catalog[catalog["display"] == selected_option].iloc[0]
            st.session_state.selected_song_id       = row["song_id"]
            st.session_state.selected_song_display  = row["display"]
            st.session_state.selected_song_mode     = row["mode"]
            st.session_state.recommendation_mode    = ""
            st.session_state.recommendations        = []
            st.session_state.refresh_offset         = 0
            with st.spinner("Finding recommendations..."):
                mode, recs = run_recommendations(row["song_id"])
            st.session_state.recommendation_mode = mode
            st.session_state.recommendations     = recs

        if st.session_state.selected_song_display:
            st.markdown("<div style='margin-top:20px;'></div>",
                        unsafe_allow_html=True)
            st.subheader("Now playing")
            st.markdown(
                f"""
                <div style="padding:20px;border-radius:12px;
                            background:{T['card_now']};
                            border:1px solid {T['border_hi']};
                            margin-bottom:8px;">
                    <div style="font-size:36px;">🎵</div>
                    <div style="font-size:20px;font-weight:bold;
                                color:{T['white']};margin-top:6px;">
                        {st.session_state.selected_song_display}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.session_state.recommendation_mode:
                st.caption(
                    f"✦ {MODE_LABELS.get(st.session_state.recommendation_mode, '')}"
                )
        else:
            st.markdown(
                f"""
                <div style="margin-top:32px;text-align:center;padding:40px 20px;
                            border:1px dashed {T['border']};
                            border-radius:12px;color:{T['muted']};">
                    <div style="font-size:48px;margin-bottom:12px;">🎵</div>
                    <div style="font-size:15px;font-weight:500;
                                color:{T['text']};margin-bottom:6px;">
                        No song selected yet
                    </div>
                    <div style="font-size:13px;">
                        Search above to get recommendations
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_right:
        st.subheader("Recommendations")

        if st.session_state.recommendations:
            _, col_refresh = st.columns([3, 1])
            with col_refresh:
                if st.button("🔄 Refresh", use_container_width=True,
                             key="single_refresh"):
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
                recommendation_card(
                    i=i, song_id=song_id,
                    seed_id=st.session_state.selected_song_id,
                    mode=st.session_state.recommendation_mode,
                    prefix="single",
                    total=len(st.session_state.recommendations),
                )
        elif st.session_state.recommendation_mode:
            st.info("No recommendations found.")
        else:
            st.markdown(
                f"<div style='text-align:center;padding:60px 20px;"
                f"color:{T['muted']};font-size:13px;'>"
                f"Recommendations appear here once you select a song</div>",
                unsafe_allow_html=True,
            )


# playlist tab

with tab_playlist:
    st.markdown(
        f"""
        <div style="background:{T['card']};border:1px solid {T['border']};
                    border-radius:12px;padding:14px 18px;margin-bottom:16px;">
            <div style="font-size:14px;color:{T['purple']};margin-bottom:4px;">
                How it works
            </div>
            <div style="font-size:13px;color:{T['muted']};">
                Build a mini playlist of up to 10 songs, then get recommendations
                based on the overall sound and listening patterns of your whole
                playlist — not just one song.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_pl_left, col_pl_right = st.columns([1, 1], gap="large")

    with col_pl_left:
        st.subheader("Build your playlist")

        pl_option = st.selectbox(
            "Add a song",
            options=catalog["display"].tolist(),
            index=None,
            placeholder="Search for a song to add...",
            label_visibility="collapsed",
            key="playlist_search",
        )

        col_add, col_clear = st.columns([1, 1])

        with col_add:
            if st.button("+ Add to playlist", use_container_width=True,
                         key="pl_add"):
                if pl_option is None:
                    st.warning("Select a song first.")
                elif len(st.session_state.playlist) >= 10:
                    st.warning("Playlist limit is 10 songs.")
                else:
                    row = catalog[catalog["display"] == pl_option].iloc[0]
                    existing = [s["song_id"] for s in st.session_state.playlist]
                    if row["song_id"] in existing:
                        st.warning("That song is already in your playlist.")
                    else:
                        st.session_state.playlist.append({
                            "song_id": row["song_id"],
                            "display": row["display"],
                        })
                        st.session_state.playlist_recs           = []
                        st.session_state.playlist_mode           = ""
                        st.session_state.playlist_excluded       = set()
                        st.session_state.playlist_refresh_offset = 0
                        st.rerun()

        with col_clear:
            if st.button("Clear playlist", use_container_width=True,
                         key="pl_clear"):
                st.session_state.playlist                = []
                st.session_state.playlist_recs           = []
                st.session_state.playlist_mode           = ""
                st.session_state.playlist_excluded       = set()
                st.session_state.playlist_refresh_offset = 0
                st.rerun()

        if st.session_state.playlist:
            st.markdown(
                f"<div style='margin-top:16px;font-size:11px;"
                f"color:{T['purple']};text-transform:uppercase;"
                f"letter-spacing:0.08em;margin-bottom:8px;'>"
                f"Your playlist ({len(st.session_state.playlist)}/10)</div>",
                unsafe_allow_html=True,
            )

            for idx, song in enumerate(st.session_state.playlist):
                col_name, col_rm = st.columns([5, 1])
                with col_name:
                    st.markdown(
                        f"""
                        <div style="background:{T['card']};
                                    border:1px solid {T['border']};
                                    border-radius:8px;padding:10px 14px;
                                    margin-bottom:4px;">
                            <span style="font-size:12px;color:{T['muted']};">
                                {idx + 1}.
                            </span>
                            <span style="font-size:14px;color:{T['white']};
                                         margin-left:6px;">
                                {song['display']}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col_rm:
                    if st.button("✕", key=f"pl_remove_{idx}_{song['song_id']}",
                                 help=f"Remove {song['display']}"):
                        st.session_state.playlist.pop(idx)
                        st.session_state.playlist_recs           = []
                        st.session_state.playlist_mode           = ""
                        st.session_state.playlist_excluded       = set()
                        st.session_state.playlist_refresh_offset = 0
                        st.rerun()

            st.markdown("<div style='margin-top:12px;'></div>",
                        unsafe_allow_html=True)

            if st.button("🎵 Recommend from playlist",
                         use_container_width=True, key="pl_recommend"):
                with st.spinner("Finding recommendations for your playlist..."):
                    mode, recs = run_playlist_recommendations(offset=0)
                st.session_state.playlist_mode           = mode
                st.session_state.playlist_recs           = recs
                st.session_state.playlist_refresh_offset = 0
                st.rerun()

        else:
            st.markdown(
                f"""
                <div style="margin-top:20px;text-align:center;padding:30px 20px;
                            border:1px dashed {T['border']};
                            border-radius:12px;color:{T['muted']};">
                    <div style="font-size:32px;margin-bottom:10px;">📋</div>
                    <div style="font-size:14px;color:{T['text']};
                                margin-bottom:4px;">
                        Your playlist is empty
                    </div>
                    <div style="font-size:13px;">
                        Search for songs above and add them
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_pl_right:
        st.subheader("Recommendations")

        if st.session_state.playlist_recs:
            _, col_refresh = st.columns([3, 1])
            with col_refresh:
                if st.button("🔄 Refresh", use_container_width=True,
                             key="pl_refresh"):
                    st.session_state.playlist_refresh_offset += NUM_RECOMMENDATIONS
                    with st.spinner("Refreshing..."):
                        mode, recs = run_playlist_recommendations(
                            offset=st.session_state.playlist_refresh_offset
                        )
                    st.session_state.playlist_mode = mode
                    st.session_state.playlist_recs = recs
                    st.rerun()

            if st.session_state.playlist_mode:
                st.caption(
                    f"✦ {MODE_LABELS.get(st.session_state.playlist_mode, '')}"
                    " — based on your full playlist"
                )

            seed_id = (st.session_state.playlist[0]["song_id"]
                       if st.session_state.playlist else None)

            for i, song_id in enumerate(st.session_state.playlist_recs):
                recommendation_card(
                    i=i, song_id=song_id,
                    seed_id=seed_id,
                    mode=st.session_state.playlist_mode,
                    prefix="playlist",
                    total=len(st.session_state.playlist_recs),
                )

        elif st.session_state.playlist_mode:
            st.info("No recommendations found.")
        elif st.session_state.playlist:
            st.markdown(
                f"<div style='text-align:center;padding:60px 20px;"
                f"color:{T['muted']};font-size:13px;'>"
                f"Click 'Recommend from playlist' to get started</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='text-align:center;padding:60px 20px;"
                f"color:{T['muted']};font-size:13px;'>"
                f"Add songs to your playlist first</div>",
                unsafe_allow_html=True,
            )


# sidebar - hidden songs

with st.sidebar:
    st.header("Hidden Songs")

    all_excluded = (st.session_state.excluded_songs
                    | st.session_state.playlist_excluded)

    if not all_excluded:
        st.caption("No songs hidden yet. Use ✕ Hide on any recommendation.")
    else:
        st.caption(f"{len(all_excluded)} song(s) hidden this session.")

        for song_id in list(all_excluded):
            row     = catalog[catalog["song_id"] == song_id]
            display = row["display"].values[0] if not row.empty else song_id.title()
            col_name, col_restore = st.columns([4, 1])

            with col_name:
                st.markdown(
                    f"<span style='font-size:13px;color:{T['muted']};'>"
                    f"{display}</span>",
                    unsafe_allow_html=True,
                )
            with col_restore:
                if st.button("↩", key=f"restore_{song_id}",
                             help=f"Unhide {display}"):
                    st.session_state.excluded_songs.discard(song_id)
                    st.session_state.playlist_excluded.discard(song_id)
                    st.rerun()

        st.markdown("---")
        if st.button("Unhide all", use_container_width=True):
            st.session_state.excluded_songs    = set()
            st.session_state.playlist_excluded = set()
            st.rerun()