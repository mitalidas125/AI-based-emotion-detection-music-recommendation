"""
🎵 AI-Based Emotion Detection & Music Recommendation System
BCA Final Year Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import time
import datetime

# Optional imports with fallback
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EmotionTunes AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

    /* Cards */
    .emotion-card {
        background: rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255,255,255,0.12);
        backdrop-filter: blur(10px);
        margin-bottom: 16px;
    }

    /* Metric boxes */
    .metric-box {
        background: rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* Big emotion display */
    .emotion-result {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        padding: 20px;
        border-radius: 16px;
        margin: 10px 0;
    }

    /* Music card */
    .music-card {
        background: rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 14px 18px;
        margin: 8px 0;
        border-left: 4px solid;
        transition: transform 0.2s;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15,12,41,0.9);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    h1, h2, h3 { color: white; }
    p, label { color: rgba(255,255,255,0.85); }
</style>
""", unsafe_allow_html=True)

# ── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load("emotion_model.pkl")
        with open("model_info.json") as f:
            info = json.load(f)
        return model, info
    except FileNotFoundError:
        return None, None

model, model_info = load_model()

# ── Data ─────────────────────────────────────────────────────────────────────
EMOTION_CONFIG = {
    "Happy": {
        "emoji": "😊",
        "color": "#FFD700",
        "gradient": "linear-gradient(135deg, #f7971e, #ffd200)",
        "description": "You seem joyful and energetic!",
        "music_genres": ["Pop", "Dance", "Upbeat Indie", "Feel-Good R&B"],
        "youtube_queries": ["happy upbeat songs 2024", "feel good pop music", "dance hits playlist"],
        "spotify_mood": "happy",
    },
    "Sad": {
        "emoji": "😢",
        "color": "#6495ED",
        "gradient": "linear-gradient(135deg, #4facfe, #00f2fe)",
        "description": "You seem to be going through a tough time.",
        "music_genres": ["Soft Acoustic", "Lo-fi", "Classical", "Soul"],
        "youtube_queries": ["sad songs playlist", "emotional acoustic music", "lofi sad beats"],
        "spotify_mood": "sad",
    },
    "Angry": {
        "emoji": "😠",
        "color": "#FF4500",
        "gradient": "linear-gradient(135deg, #f12711, #f5af19)",
        "description": "You seem frustrated or intense right now.",
        "music_genres": ["Rock", "Metal", "Rap", "Intense EDM"],
        "youtube_queries": ["rock music workout", "intense rap playlist", "heavy metal hits"],
        "spotify_mood": "angry",
    },
    "Calm": {
        "emoji": "😌",
        "color": "#90EE90",
        "gradient": "linear-gradient(135deg, #11998e, #38ef7d)",
        "description": "You seem peaceful and relaxed.",
        "music_genres": ["Ambient", "Jazz", "Meditation", "Soft Classical"],
        "youtube_queries": ["calm relaxing music", "ambient meditation music", "peaceful jazz"],
        "spotify_mood": "calm",
    },
}

FEATURES = [
    "typing_speed_wpm", "keypress_duration_ms", "pause_duration_ms",
    "error_rate", "backspace_count", "avg_word_length",
    "sentence_length", "exclamation_freq"
]

# ── Session State ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "keystrokes" not in st.session_state:
    st.session_state.keystrokes = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎵 EmotionTunes AI")
    st.markdown("---")
    st.markdown("### 📖 About")
    st.info(
        "This system detects your emotional state by analysing **how** you type — "
        "speed, pauses, errors, and patterns — then recommends matching music."
    )

    st.markdown("### 🧠 How It Works")
    st.markdown("""
    1. You type any text below
    2. AI analyses your typing behaviour
    3. Emotion is predicted (Happy/Sad/Angry/Calm)
    4. Music is recommended based on emotion
    """)

    st.markdown("### 📊 Model Info")
    if model_info:
        st.metric("Best Model", model_info["best_model"])
        st.metric("Accuracy", f"{model_info['best_accuracy']*100:.1f}%")
        st.metric("Training Samples", "1,200")

    st.markdown("---")
    st.markdown("### 🔬 Tech Stack")
    st.markdown("""
    - **Python** + **Streamlit**
    - **Scikit-learn** (ML)
    - **TextBlob** (NLP)
    - **Plotly** (Charts)
    - **Random Forest / SVM**
    """)

    st.markdown("---")
    st.caption("BCA Final Year Project | 2023-26")

# ── Main Content ──────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; color:white; font-size:2.8rem;'>
    🎵 EmotionTunes AI
</h1>
<p style='text-align:center; color:rgba(255,255,255,0.6); font-size:1.1rem; margin-top:-10px;'>
    Detect your emotion through typing • Get personalized music
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🤖 Auto Detect", "🎯 Manual Mode", "📊 Analytics", "🆚 Model Comparison", "ℹ️ About"])

# ═══════════════════════════════════════════════════════
# TAB 1: AUTO DETECT (HTML embedded)
# ═══════════════════════════════════════════════════════
with tab1:
    st.markdown("### 🤖 Auto Detect — Just Type & Get Result!")
    st.info("👇 Type below — everything is measured automatically — no sliders needed!")

    # Read and embed the HTML file
    html_content = open("EmotionTunes_LiveDetector.html", "r", encoding="utf-8").read() if __import__("os").path.exists("EmotionTunes_LiveDetector.html") else None

    if html_content:
        import streamlit.components.v1 as components
        components.html(html_content, height=900, scrolling=True)
    else:
        st.warning("⚠️ EmotionTunes_LiveDetector.html file not found! Please place it in the same folder as app.py")
        st.markdown("""
        **How to fix:**
        1. Download the `EmotionTunes_LiveDetector.html` file
        2. Place it in same folder as `app.py`
        3. Restart the app with `streamlit run app.py`
        """)

# ═══════════════════════════════════════════════════════

# TAB 2: MANUAL MODE (was tab2)
# ═══════════════════════════════════════════════════════
with tab2:
    st.markdown("### ✍️ Type anything below...")
    st.caption("Type naturally — the AI analyses your typing patterns, not just your words.")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Text input
        user_text = st.text_area(
            "Your text:",
            height=150,
            placeholder="Start typing here... (write at least 20 words for best accuracy)",
            label_visibility="collapsed"
        )

        # Typing speed input (since we can't capture real-time in Streamlit easily)
        st.markdown("#### ⚙️ Typing Parameters")
        st.caption("Adjust these based on how you typed the text above")

        c1, c2, c3 = st.columns(3)
        with c1:
            typing_speed = st.slider("Typing Speed (WPM)", 10, 150, 55,
                                     help="Words per minute — fast=angry/happy, slow=sad/calm")
        with c2:
            keypress_dur = st.slider("Key Hold Duration (ms)", 30, 300, 95,
                                     help="How long each key is held — longer = sad/calm")
        with c3:
            pause_dur = st.slider("Pause Between Words (ms)", 50, 800, 200,
                                  help="Pause time between words")

        c4, c5, c6 = st.columns(3)
        with c4:
            error_rate = st.slider("Error Rate", 0.0, 0.30, 0.03, step=0.01,
                                   help="Fraction of keys that were errors (0 = no errors)")
        with c5:
            backspace_count = st.slider("Backspace Count", 0, 20, 2,
                                        help="How many times you pressed backspace")
        with c6:
            exclamation_freq = st.slider("Exclamation Freq", 0.0, 1.0, 0.1, step=0.05,
                                         help="How often you used ! in your text")

    with col2:
        st.markdown("#### 💡 Quick Presets")
        st.caption("Tap a preset to auto-fill sliders:")

        if st.button("😊 I'm Happy", use_container_width=True):
            st.session_state["preset"] = "Happy"
            st.rerun()
        if st.button("😢 I'm Sad", use_container_width=True):
            st.session_state["preset"] = "Sad"
            st.rerun()
        if st.button("😠 I'm Angry", use_container_width=True):
            st.session_state["preset"] = "Angry"
            st.rerun()
        if st.button("😌 I'm Calm", use_container_width=True):
            st.session_state["preset"] = "Calm"
            st.rerun()

        # Apply preset if selected
        preset = st.session_state.get("preset")
        if preset == "Happy":
            typing_speed, keypress_dur, pause_dur = 65, 80, 150
            error_rate, backspace_count, exclamation_freq = 0.03, 2, 0.6
        elif preset == "Sad":
            typing_speed, keypress_dur, pause_dur = 38, 130, 400
            error_rate, backspace_count, exclamation_freq = 0.05, 4, 0.05
        elif preset == "Angry":
            typing_speed, keypress_dur, pause_dur = 85, 60, 80
            error_rate, backspace_count, exclamation_freq = 0.10, 7, 0.8
        elif preset == "Calm":
            typing_speed, keypress_dur, pause_dur = 55, 95, 200
            error_rate, backspace_count, exclamation_freq = 0.02, 2, 0.1

        st.markdown("---")
        st.markdown("#### 📈 Live Stats")
        if user_text:
            words = len(user_text.split())
            chars = len(user_text)
            st.metric("Words", words)
            st.metric("Characters", chars)
            sentences = user_text.count('.') + user_text.count('!') + user_text.count('?')
            st.metric("Sentences", max(1, sentences))

    st.markdown("---")

    # ── ANALYZE BUTTON ──
    analyze_btn = st.button("🔍 Analyze My Emotion", type="primary", use_container_width=True)

    if analyze_btn:
        if not user_text or len(user_text.split()) < 3:
            st.warning("⚠️ Please type at least a few words for analysis!")
        elif model is None:
            st.error("❌ Model not found! Please run `python train_model.py` first.")
        else:
            with st.spinner("🧠 Analysing your typing patterns..."):
                time.sleep(0.8)  # Small delay for effect

                # Prepare features
                words = user_text.split()
                avg_word_len = np.mean([len(w) for w in words]) if words else 5.0
                sentence_len = len(words) / max(1, user_text.count('.') + user_text.count('!') + user_text.count('?') + 1)

                features = np.array([[
                    typing_speed,
                    keypress_dur,
                    pause_dur,
                    error_rate,
                    backspace_count,
                    avg_word_len,
                    sentence_len,
                    exclamation_freq
                ]])

                # Predict
                emotion = model.predict(features)[0]
                proba = model.predict_proba(features)[0]
                classes = model.classes_
                confidence = max(proba) * 100

                # Sentiment analysis
                if HAS_TEXTBLOB:
                    blob = TextBlob(user_text)
                    sentiment_score = blob.sentiment.polarity
                    subjectivity   = blob.sentiment.subjectivity
                else:
                    # Simple rule-based fallback
                    positive_words = ["happy","great","love","good","wonderful","amazing","joy","excited","beautiful","fantastic"]
                    negative_words = ["sad","bad","hate","terrible","awful","angry","upset","depressed","miserable","horrible"]
                    words_lower = user_text.lower().split()
                    pos = sum(1 for w in words_lower if w in positive_words)
                    neg = sum(1 for w in words_lower if w in negative_words)
                    sentiment_score = (pos - neg) / max(1, len(words_lower))
                    subjectivity = min(1.0, (pos + neg) / max(1, len(words_lower)) * 5)

                if sentiment_score > 0.05:
                    sentiment_label = "Positive 😊"
                    sent_color = "#90EE90"
                elif sentiment_score < -0.05:
                    sentiment_label = "Negative 😔"
                    sent_color = "#FF6B6B"
                else:
                    sentiment_label = "Neutral 😐"
                    sent_color = "#87CEEB"

                # Save to history
                st.session_state.history.append({
                    "time"      : datetime.datetime.now().strftime("%H:%M:%S"),
                    "emotion"   : emotion,
                    "confidence": round(confidence, 1),
                    "sentiment" : sentiment_label,
                    "text_preview": user_text[:40] + "..." if len(user_text) > 40 else user_text
                })

            # ── RESULTS ──
            config = EMOTION_CONFIG[emotion]

            st.markdown("---")
            st.markdown("## 🎯 Detection Result")

            r1, r2, r3 = st.columns(3)

            with r1:
                st.markdown(f"""
                <div class="emotion-card" style="background:{config['gradient']}20; border-color:{config['color']}44;">
                    <div class="emotion-result" style="background:{config['gradient']}; -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                        {config['emoji']} {emotion}
                    </div>
                    <p style="text-align:center; color:rgba(255,255,255,0.8);">{config['description']}</p>
                </div>
                """, unsafe_allow_html=True)

            with r2:
                # Confidence display
                if HAS_PLOTLY:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence,
                        title={"text": "Confidence %", "font": {"color": "white"}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "white"},
                            "bar" : {"color": config["color"]},
                            "bgcolor": "rgba(255,255,255,0.1)",
                            "steps": [
                                {"range": [0, 50], "color": "rgba(255,0,0,0.1)"},
                                {"range": [50, 75], "color": "rgba(255,165,0,0.1)"},
                                {"range": [75, 100], "color": "rgba(0,255,0,0.1)"},
                            ],
                        },
                        number={"suffix": "%", "font": {"color": "white", "size": 28}},
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=200,
                        margin=dict(l=20, r=20, t=40, b=10),
                        font={"color": "white"}
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)
                else:
                    st.markdown(f"""
                    <div class="emotion-card" style="text-align:center;">
                        <h4 style="color:white;">🎯 Confidence</h4>
                        <p style="font-size:3rem; font-weight:800; color:{config['color']};">{confidence:.1f}%</p>
                        <p style="color:rgba(255,255,255,0.6);">Model confidence</p>
                    </div>
                    """, unsafe_allow_html=True)

            with r3:
                # Sentiment Analysis result
                st.markdown(f"""
                <div class="emotion-card">
                    <h4 style="color:white;">📝 Text Sentiment</h4>
                    <p style="color:{sent_color}; font-size:1.3rem; font-weight:700;">{sentiment_label}</p>
                    <p style="color:rgba(255,255,255,0.7);">Polarity: {sentiment_score:.2f}</p>
                    <p style="color:rgba(255,255,255,0.7);">Subjectivity: {subjectivity:.2f}</p>
                    <hr style="border-color:rgba(255,255,255,0.2);">
                    <p style="color:rgba(255,255,255,0.5); font-size:0.85rem;">
                        Sentiment: analysis of your actual words<br>
                        Emotion: analysis of your typing behaviour
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # ── Probability breakdown ──
            st.markdown("#### 📊 Emotion Probability Breakdown")
            prob_df = pd.DataFrame({
                "Emotion": [f"{EMOTION_CONFIG[c]['emoji']} {c}" for c in classes],
                "Probability": [round(p * 100, 1) for p in proba],
            })
            if HAS_PLOTLY:
                fig_bar = px.bar(
                    prob_df, x="Emotion", y="Probability",
                    color="Emotion",
                    color_discrete_map={f"{EMOTION_CONFIG[c]['emoji']} {c}": EMOTION_CONFIG[c]["color"] for c in classes},
                    text="Probability",
                )
                fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    yaxis=dict(range=[0, 110], color="white"),
                    xaxis=dict(color="white"),
                    font={"color": "white"},
                    height=280,
                    margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.bar_chart(prob_df.set_index("Emotion"))

            # ── MUSIC RECOMMENDATIONS ──
            st.markdown("---")
            st.markdown(f"## 🎵 Music Recommendations for **{emotion}** mood")

            m1, m2 = st.columns([1, 1])
            with m1:
                st.markdown("#### 🎸 Recommended Genres")
                for genre in config["music_genres"]:
                    st.markdown(f"""
                    <div class="music-card" style="border-color:{config['color']};">
                        <span style="color:white; font-weight:600;">🎵 {genre}</span>
                    </div>
                    """, unsafe_allow_html=True)

            with m2:
                st.markdown("#### 🔗 Listen Now")
                for query in config["youtube_queries"]:
                    yt_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
                    st.markdown(f"""
                    <a href="{yt_url}" target="_blank" style="text-decoration:none;">
                        <div class="music-card" style="border-color:#FF0000; cursor:pointer;">
                            <span style="color:white;">▶ {query.title()}</span>
                            <span style="float:right; color:#FF0000; font-size:0.8rem;">YouTube →</span>
                        </div>
                    </a>
                    """, unsafe_allow_html=True)

                # Spotify search link
                spotify_url = f"https://open.spotify.com/search/{config['spotify_mood']}%20music"
                st.markdown(f"""
                <a href="{spotify_url}" target="_blank" style="text-decoration:none;">
                    <div class="music-card" style="border-color:#1DB954; cursor:pointer; margin-top:12px;">
                        <span style="color:white;">🎧 Open in Spotify</span>
                        <span style="float:right; color:#1DB954; font-size:0.8rem;">Spotify →</span>
                    </div>
                </a>
                """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# TAB 3: ANALYTICS
# ═══════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Your Emotion History")

    if not st.session_state.history:
        st.info("🎯 No history yet! Go to 'Detect Emotion' tab and analyse some text first.")
    else:
        history_df = pd.DataFrame(st.session_state.history)

        # Summary metrics
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            st.metric("Total Analyses", len(history_df))
        with a2:
            dominant = history_df["emotion"].mode()[0]
            st.metric("Dominant Emotion", f"{EMOTION_CONFIG[dominant]['emoji']} {dominant}")
        with a3:
            avg_conf = history_df["confidence"].mean()
            st.metric("Avg Confidence", f"{avg_conf:.1f}%")
        with a4:
            unique_emotions = history_df["emotion"].nunique()
            st.metric("Emotions Detected", f"{unique_emotions}/4")

        # Emotion frequency pie chart
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Emotion Distribution")
            emotion_counts = history_df["emotion"].value_counts()
            if HAS_PLOTLY:
                fig_pie = px.pie(
                    values=emotion_counts.values,
                    names=[f"{EMOTION_CONFIG[e]['emoji']} {e}" for e in emotion_counts.index],
                    color_discrete_sequence=[EMOTION_CONFIG[e]["color"] for e in emotion_counts.index],
                    hole=0.4,
                )
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"color": "white"},
                    legend=dict(font=dict(color="white")),
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.bar_chart(emotion_counts)

        with col_b:
            st.markdown("#### Confidence Over Time")
            if HAS_PLOTLY:
                fig_line = px.line(
                    history_df,
                    x=list(range(1, len(history_df)+1)),
                    y="confidence",
                    color="emotion",
                    color_discrete_map={e: EMOTION_CONFIG[e]["color"] for e in EMOTION_CONFIG},
                    markers=True,
                    labels={"x": "Analysis #", "confidence": "Confidence %"},
                )
                fig_line.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "white"},
                    xaxis=dict(color="white"),
                    yaxis=dict(color="white", range=[0, 105]),
                )
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.line_chart(history_df.set_index("time")["confidence"])

        # History table
        st.markdown("#### 📋 Detailed History")
        display_df = history_df.copy()
        display_df["emotion"] = display_df["emotion"].apply(
            lambda e: f"{EMOTION_CONFIG[e]['emoji']} {e}"
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

# ═══════════════════════════════════════════════════════
# TAB 4: MODEL COMPARISON
# ═══════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🆚 ML Model Comparison")
    st.caption("We trained 4 different models and selected the best one automatically.")

    if model_info:
        comparison = model_info.get("comparison", {})

        comp_data = []
        for name, metrics in comparison.items():
            comp_data.append({
                "Model": name,
                "Test Accuracy": f"{metrics['test_accuracy']*100:.2f}%",
                "CV Mean Accuracy": f"{metrics['cv_mean']*100:.2f}%",
                "CV Std Dev": f"±{metrics['cv_std']*100:.2f}%",
                "Best?": "🏆 Yes" if name == model_info["best_model"] else "—"
            })

        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Bar chart comparison
        names = list(comparison.keys())
        test_accs = [comparison[n]["test_accuracy"]*100 for n in names]
        cv_accs   = [comparison[n]["cv_mean"]*100 for n in names]

        if HAS_PLOTLY:
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(name="Test Accuracy", x=names, y=test_accs,
                                       marker_color="#6C63FF", text=[f"{v:.1f}%" for v in test_accs],
                                       textposition="outside"))
            fig_comp.add_trace(go.Bar(name="CV Mean Accuracy", x=names, y=cv_accs,
                                       marker_color="#FF6584", text=[f"{v:.1f}%" for v in cv_accs],
                                       textposition="outside"))
            fig_comp.update_layout(
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
                yaxis=dict(range=[0, 115], color="white"),
                xaxis=dict(color="white"),
                legend=dict(font=dict(color="white")),
                height=380,
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            chart_data = pd.DataFrame({"Test Accuracy": test_accs, "CV Mean": cv_accs}, index=names)
            st.bar_chart(chart_data)

        # Feature importance
        if model_info.get("feature_importance"):
            st.markdown("#### 🔑 Feature Importance (Top Features)")
            fi = model_info["feature_importance"]
            fi_sorted = sorted(fi.items(), key=lambda x: -x[1])
            fi_names  = [x[0].replace("_", " ").title() for x in fi_sorted]
            fi_vals   = [x[1] for x in fi_sorted]

            if HAS_PLOTLY:
                fig_fi = px.bar(
                    x=fi_vals, y=fi_names, orientation="h",
                    labels={"x": "Importance", "y": "Feature"},
                    color=fi_vals,
                    color_continuous_scale="viridis",
                )
                fig_fi.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "white"},
                    yaxis=dict(color="white"),
                    xaxis=dict(color="white"),
                    coloraxis_showscale=False,
                    height=340,
                )
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                fi_df = pd.DataFrame({"Importance": fi_vals}, index=fi_names)
                st.bar_chart(fi_df)
    else:
        st.warning("Model info not found. Please run `python train_model.py` first.")

# ═══════════════════════════════════════════════════════
# TAB 5: ABOUT
# ═══════════════════════════════════════════════════════
with tab5:
    st.markdown("### ℹ️ About This Project")

    a1, a2 = st.columns(2)

    with a1:
        st.markdown("""
        <div class="emotion-card">
            <h4>🎯 Project Goal</h4>
            <p>To build an AI system that understands human emotions through typing behaviour 
            and recommends suitable music to match or improve the user's mood.</p>
        </div>

        <div class="emotion-card">
            <h4>🔬 Methodology</h4>
            <p><b>1. Data Collection:</b> Typing features are extracted — speed, pauses, errors, 
            key hold duration, and text patterns.</p>
            <p><b>2. NLP Analysis:</b> TextBlob performs sentiment analysis on the typed text 
            to supplement behavioural features.</p>
            <p><b>3. ML Classification:</b> 4 models trained and compared. Best model selected 
            automatically based on cross-validation accuracy.</p>
            <p><b>4. Recommendation:</b> Music genres and direct links provided based on 
            detected emotion.</p>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div class="emotion-card">
            <h4>⚠️ Limitations</h4>
            <p>• Typing speed input is manual (real-time keystroke capture needs JavaScript)</p>
            <p>• Small dataset — may not generalise to all users equally</p>
            <p>• Emotion detection is probabilistic, not definitive</p>
            <p>• Music recommendations are category-based, not personalised</p>
        </div>

        <div class="emotion-card">
            <h4>🔮 Future Scope</h4>
            <p>• Real-time keystroke capture using JavaScript + Streamlit components</p>
            <p>• Facial emotion detection using webcam (OpenCV + DeepFace)</p>
            <p>• Spotify API integration for actual song recommendations</p>
            <p>• User feedback loop for model improvement</p>
            <p>• Mobile app version</p>
            <p>• Personalised recommendations using collaborative filtering</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="emotion-card" style="text-align:center;">
        <h4>👨‍💻 Project Details</h4>
        <p><b>Course:</b> Bachelor of Computer Applications (BCA) — Final Year Project</p>
        <p><b>Technologies:</b> Python, Streamlit, Scikit-learn, TextBlob, Plotly, Pandas</p>
        <p><b>ML Algorithms:</b> Random Forest, SVM, Gradient Boosting, Logistic Regression</p>
    </div>
    """, unsafe_allow_html=True)