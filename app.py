import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Music Rec System",
    page_icon="🎵",
    layout="wide",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0d0d;
    color: #f0f0f0;
}

h1, h2, h3 { font-family: 'Space Mono', monospace; }

.stApp { background-color: #0d0d0d; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 1px solid #2a2a2a;
}

/* Inputs */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stSlider > div {
    background-color: #1a1a1a !important;
    color: #f0f0f0 !important;
    border: 1px solid #333 !important;
    border-radius: 6px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1db954, #1aa34a);
    color: #000;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    border: none;
    border-radius: 6px;
    padding: 0.6rem 1.8rem;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(29,185,84,0.35);
}

/* Metric cards */
.metric-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-card .label {
    font-size: 0.7rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'Space Mono', monospace;
}
.metric-card .value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #1db954;
    font-family: 'Space Mono', monospace;
}

/* Song rows */
.song-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.75rem 1rem;
    border-radius: 8px;
    border: 1px solid #1e1e1e;
    margin-bottom: 0.4rem;
    background: #141414;
    transition: background 0.15s;
}
.song-row:hover { background: #1e1e1e; }
.song-num {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #555;
    min-width: 24px;
}
.song-title { font-weight: 600; font-size: 0.95rem; }
.song-artist { font-size: 0.8rem; color: #888; }
.song-score {
    margin-left: auto;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #1db954;
}
.emotion-badge {
    font-size: 0.65rem;
    font-family: 'Space Mono', monospace;
    padding: 2px 8px;
    border-radius: 20px;
    background: #1e1e1e;
    border: 1px solid #333;
    color: #aaa;
    white-space: nowrap;
}

/* Header banner */
.header-banner {
    padding: 2rem 0 1rem 0;
    border-bottom: 1px solid #222;
    margin-bottom: 2rem;
}
.header-banner h1 { font-size: 2rem; margin: 0; color: #f0f0f0; }
.header-banner p { color: #666; font-size: 0.9rem; margin: 0.3rem 0 0 0; }

/* Section headers */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #555;
    margin-bottom: 0.75rem;
}

/* Arc tag */
.arc-tag {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 3px 10px;
    border-radius: 4px;
    background: #1a2e1a;
    border: 1px solid #1db954;
    color: #1db954;
    margin-bottom: 1.5rem;
}

/* Dataframe tweaks */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* Divider */
hr { border-color: #222 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# EMOTION HELPERS (must match notebook)
# ─────────────────────────────────────────────
EMOTIONAL_ARCS = {
    "chill_to_hype": ["sad_calm", "happy_calm", "happy_energetic"],
    "hype_to_chill": ["happy_energetic", "happy_calm", "sad_calm"],
}

EMOTION_EMOJI = {
    "happy_energetic": "🔥",
    "happy_calm":      "😊",
    "sad_calm":        "🌙",
    "intense":         "⚡",
}

def get_emotion(row):
    if row["valence"] > 0.6 and row["energy"] > 0.6:
        return "happy_energetic"
    elif row["valence"] > 0.6:
        return "happy_calm"
    elif row["energy"] > 0.6:
        return "intense"
    else:
        return "sad_calm"

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def apply_emotional_arc(df_subset, arc_name="chill_to_hype"):
    arc = EMOTIONAL_ARCS[arc_name]
    emotion_rank = {e: i for i, e in enumerate(arc)}
    df_subset = df_subset.copy()
    df_subset["emotion_rank"] = df_subset["emotion"].map(
        lambda x: emotion_rank.get(x, len(arc))
    )
    return df_subset.sort_values(["emotion_rank", "energy"])

# ─────────────────────────────────────────────
# DATA + MODEL LOADING (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="🎵 Building models — this only runs once...")
def load_and_build():
    df = pd.read_csv("Audio_featuresandlyrics.csv")

    # Clean
    df = df[['track_id','track_name','track_artist','lyrics','track_popularity',
             'track_album_name','track_album_release_date','playlist_genre',
             'playlist_subgenre','danceability','energy','key','loudness','mode',
             'speechiness','acousticness','instrumentalness','liveness','valence',
             'tempo','duration_ms','language']]
    df = df.loc[(df['language'] == 'en') & (df['playlist_genre'] != 'latin')]
    df = df.loc[df['lyrics'] != 'Lyrics for this song have yet to be released. Please check back once the song has been released.']
    df = df.reset_index(drop=True)

    # Lyrics
    df["clean_lyrics"] = df["lyrics"].apply(clean_text)

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df["clean_lyrics"])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Clusters
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(tfidf_matrix)

    # Emotions
    df["emotion"] = df.apply(get_emotion, axis=1)

    # Audio features
    audio_features = ["danceability","energy","key","loudness","mode",
                      "speechiness","acousticness","instrumentalness",
                      "liveness","valence","tempo","duration_ms"]
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[audio_features] = scaler.fit_transform(df[audio_features])
    feature_matrix = df_scaled[audio_features].values
    similarity_matrixA = cosine_similarity(feature_matrix)

    return df, similarity_matrix, similarity_matrixA

# ─────────────────────────────────────────────
# RECOMMENDATION FUNCTIONS
# ─────────────────────────────────────────────
def get_song_index(song_name, df):
    matches = df[df["track_name"].str.lower() == song_name.lower()]
    return matches.index[0] if len(matches) > 0 else None

def generate_playlist(song_name, df, similarity_matrix, top_n=15, arc="chill_to_hype"):
    song_idx = get_song_index(song_name, df)
    if song_idx is None:
        return None

    cluster_id = df.loc[song_idx, "cluster"]
    cluster_df = df[df["cluster"] == cluster_id].copy()
    cluster_indices = cluster_df.index

    scores = similarity_matrix[song_idx][cluster_indices]
    cluster_df["lyric_sim"] = scores
    cluster_df = cluster_df[cluster_df.index != song_idx]
    cluster_df = cluster_df.sort_values("lyric_sim", ascending=False).head(top_n * 3)
    cluster_df = apply_emotional_arc(cluster_df, arc_name=arc)
    return cluster_df.head(top_n)

def get_similar_songs_AudF(song_name, df, similarity_matrixA, top_n=15, arc="chill_to_hype"):
    song_idx = get_song_index(song_name, df)
    if song_idx is None:
        return None

    sim_scores = list(enumerate(similarity_matrixA[song_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != song_idx][:top_n * 3]

    song_indices = [i[0] for i in sim_scores]
    scores_vals  = [i[1] for i in sim_scores]

    result = df.iloc[song_indices].copy()
    result["audio_sim"] = scores_vals
    result = apply_emotional_arc(result, arc_name=arc)
    return result.head(top_n)

def generate_hybrid_playlist(song_name, df, similarity_matrix, similarity_matrixA,
                              top_n=15, arc="chill_to_hype",
                              lyric_weight=0.5, audio_weight=0.5):
    song_idx = get_song_index(song_name, df)
    if song_idx is None:
        return None

    cluster_id = df.loc[song_idx, "cluster"]
    cluster_df = df[df["cluster"] == cluster_id].copy()
    cluster_indices = cluster_df.index

    cluster_df["lyric_sim"] = similarity_matrix[song_idx][cluster_indices]
    cluster_df["audio_sim"] = similarity_matrixA[song_idx][cluster_indices]
    cluster_df["hybrid_score"] = (lyric_weight * cluster_df["lyric_sim"] +
                                   audio_weight * cluster_df["audio_sim"])

    cluster_df = cluster_df[cluster_df.index != song_idx]
    candidates = cluster_df.sort_values("hybrid_score", ascending=False).head(top_n * 3)
    candidates = apply_emotional_arc(candidates, arc_name=arc)
    return candidates.head(top_n)

# ─────────────────────────────────────────────
# RENDER PLAYLIST
# ─────────────────────────────────────────────
def render_playlist(results, mode):
    if results is None or results.empty:
        st.error("Song not found. Try a different title.")
        return

    for i, (_, row) in enumerate(results.iterrows(), 1):
        emotion_label = f"{EMOTION_EMOJI.get(row['emotion'], '🎵')} {row['emotion']}"

        if mode == "Lyrical":
            score_str = f"lyric: {row.get('lyric_sim', 0):.2f}"
        elif mode == "Audio":
            score_str = f"audio: {row.get('audio_sim', 0):.2f}"
        else:
            score_str = f"hybrid: {row.get('hybrid_score', 0):.2f}"

        st.markdown(f"""
        <div class="song-row">
            <span class="song-num">{i:02d}</span>
            <div>
                <div class="song-title">{row['track_name']}</div>
                <div class="song-artist">{row['track_artist']}</div>
            </div>
            <span class="emotion-badge">{emotion_label}</span>
            <span class="song-score">{score_str}</span>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    # Load
    df, similarity_matrix, similarity_matrixA = load_and_build()
    song_list = sorted(df["track_name"].unique().tolist())

    # ── Header ──
    st.markdown("""
    <div class="header-banner">
        <h1>🎵 Music Rec System</h1>
        <p>Lyrical similarity · Thematic coherence · Emotional arc · Audio features</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar Controls ──
    with st.sidebar:
        st.markdown("### Controls")
        st.markdown("---")

        song_input = st.text_input(
    "Seed Song",
    placeholder="e.g. Blinding Lights",
    help="Type the exact song title"
)

        mode = st.selectbox(
            "Recommendation Mode",
            ["Hybrid", "Lyrical", "Audio"],
            help="Which signals to use"
        )

        arc = st.selectbox(
            "Emotional Arc",
            ["chill_to_hype", "hype_to_chill"],
        )

        top_n = st.slider("Number of Songs", 5, 30, 15)

        if mode == "Hybrid":
            st.markdown("---")
            st.markdown("**Weight Balance**")
            lyric_w = st.slider("Lyrical Weight", 0.0, 1.0, 0.5, 0.05)
            audio_w = round(1.0 - lyric_w, 2)
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;font-family:'Space Mono',monospace;font-size:0.75rem;color:#888;margin-top:0.3rem;">
                <span>🎤 Lyrics: {lyric_w}</span>
                <span>🎸 Audio: {audio_w}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            lyric_w, audio_w = 0.5, 0.5

        st.markdown("---")
        run_btn = st.button("Generate Playlist ▶", use_container_width=True)

    # ── Main Content ──
    if not run_btn:
        # Landing state
        c1, c2, c3 = st.columns(3)
        for col, title, desc in [
            (c1, "Lyrical", "TF-IDF cosine similarity on cleaned lyrics with bigrams"),
            (c2, "Audio", "Cosine similarity on 12 normalized audio features"),
            (c3, "Hybrid", "Weighted blend of both signals + thematic clustering"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">{title}</div>
                    <div style="font-size:0.85rem;color:#aaa;margin-top:0.5rem;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("Select a seed song and hit **Generate Playlist** to get started.")
        return

    if not song_input:
        st.warning("Please select a seed song from the sidebar.")
        return

    # ── Generate ──
    with st.spinner("Building your playlist..."):
        if mode == "Lyrical":
            results = generate_playlist(song_input, df, similarity_matrix, top_n, arc)
        elif mode == "Audio":
            results = get_similar_songs_AudF(song_input, df, similarity_matrixA, top_n, arc)
        else:
            results = generate_hybrid_playlist(
                song_input, df, similarity_matrix, similarity_matrixA,
                top_n=top_n, arc=arc, lyric_weight=lyric_w, audio_weight=audio_w
            )

    if results is None:
        st.error(f"❌ '{song_input}' not found in the dataset.")
        return

    # ── Stats row ──
    st.markdown("---")
    seed_row = df[df["track_name"].str.lower() == song_input.lower()].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in [
        (c1, "Seed Song", seed_row["track_name"][:18] + "…" if len(seed_row["track_name"]) > 18 else seed_row["track_name"]),
        (c2, "Mode", mode),
        (c3, "Songs", str(len(results))),
        (c4, "Arc", "↑ Chill→Hype" if arc == "chill_to_hype" else "↓ Hype→Chill"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value" style="font-size:1rem;margin-top:0.3rem;">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Playlist ──
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-label">Your Playlist</div>', unsafe_allow_html=True)
        arc_arrow = "🌙 → 🔥" if arc == "chill_to_hype" else "🔥 → 🌙"
        st.markdown(f'<div class="arc-tag">{arc_arrow} {arc}</div>', unsafe_allow_html=True)
        render_playlist(results, mode)

    with col_right:
        st.markdown('<div class="section-label">Emotion Distribution</div>', unsafe_allow_html=True)
        emotion_counts = results["emotion"].value_counts().reset_index()
        emotion_counts.columns = ["Emotion", "Count"]
        st.dataframe(emotion_counts, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Genre Mix</div>', unsafe_allow_html=True)
        genre_counts = results["playlist_genre"].value_counts().reset_index()
        genre_counts.columns = ["Genre", "Count"]
        st.dataframe(genre_counts, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Audio Profile (avg)</div>', unsafe_allow_html=True)
        audio_cols = ["danceability", "energy", "valence", "acousticness", "speechiness"]
        audio_avg = results[audio_cols].mean().reset_index()
        audio_avg.columns = ["Feature", "Avg"]
        audio_avg["Avg"] = audio_avg["Avg"].round(3)
        st.dataframe(audio_avg, use_container_width=True, hide_index=True)

    # ── Raw data toggle ──
    with st.expander("Raw playlist data"):
        cols_to_show = ["track_name", "track_artist", "playlist_genre", "emotion", "energy", "valence"]
        if "lyric_sim" in results.columns:   cols_to_show.append("lyric_sim")
        if "audio_sim" in results.columns:   cols_to_show.append("audio_sim")
        if "hybrid_score" in results.columns: cols_to_show.append("hybrid_score")
        st.dataframe(results[cols_to_show].reset_index(drop=True), use_container_width=True)

if __name__ == "__main__":
    main()
