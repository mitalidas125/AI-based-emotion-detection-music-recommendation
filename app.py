import streamlit as st
import time
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ------------------ FILES ------------------
DATA_FILE = "typing_data.csv"
MODEL_FILE = "emotion_model.pkl"

# ------------------ APP CONFIG ------------------
st.set_page_config(page_title="Emotion Detection from Typing & Music Recommender", layout="centered")

# ------------------ HELPERS ------------------
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        cols = [
            "text_length",
            "duration_sec",
            "typing_speed_cps",
            "avg_char_interval",
            "emotion"
        ]
        return pd.DataFrame(columns=cols)

def save_sample(features: dict):
    df = load_data()
    df = pd.concat([df, pd.DataFrame([features])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

def extract_features(text: str, duration_sec: float, emotion: str | None = None):
    text = text or ""
    duration_sec = max(duration_sec, 0.001)
    length = len(text)

    typing_speed = length / duration_sec  # chars per second
    avg_interval = duration_sec / max(length, 1)

    features = {
        "text_length": length,
        "duration_sec": duration_sec,
        "typing_speed_cps": typing_speed,
        "avg_char_interval": avg_interval,
    }
    if emotion is not None:
        features["emotion"] = emotion
    return features

def train_model():
    df = load_data()
    if df.empty or df["emotion"].nunique() < 2:
        return None, "Need data from at least 2 different emotions to train the model."

    X = df[["text_length", "duration_sec", "typing_speed_cps", "avg_char_interval"]]
    y = df["emotion"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    joblib.dump(model, MODEL_FILE)
    return {"accuracy": acc, "report": report}, None

def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    return joblib.load(MODEL_FILE)

# Simple music library – change these to your favourite songs / YouTube links
MUSIC_LIBRARY = {
    "Happy": [
        ("Happy – Pharrell Williams", "https://www.youtube.com/watch?v=ZbZSe6N_BXs"),
        ("On Top Of The World – Imagine Dragons", "https://www.youtube.com/watch?v=w5tWYmIOWGk"),
    ],
    "Sad": [
        ("Let Her Go – Passenger", "https://www.youtube.com/watch?v=RBumgq5yVrA"),
        ("Someone Like You – Adele", "https://www.youtube.com/watch?v=hLQl3WQQoQ0"),
    ],
    "Angry": [
        ("Numb – Linkin Park", "https://www.youtube.com/watch?v=kXYiU_JCYtU"),
        ("Stronger – Kanye West", "https://www.youtube.com/watch?v=PsO6ZnUZI0g"),
    ],
    "Calm": [
        ("Weightless – Marconi Union", "https://www.youtube.com/watch?v=UfcAVejslrU"),
        ("River Flows In You – Yiruma", "https://www.youtube.com/watch?v=7maJOI3QMu0"),
    ],
}

# ------------------ UI STATE ------------------
if "start_time" not in st.session_state:
    st.session_state.start_time = None

if "mode" not in st.session_state:
    st.session_state.mode = "Collect Typing Samples"

# ------------------ SIDEBAR ------------------
st.sidebar.title("Navigation")
mode = st.sidebar.radio(
    "Choose mode:",
    ["Collect Typing Samples", "Train Model", "Emotion Detection Demo"],
    index=["Collect Typing Samples", "Train Model", "Emotion Detection Demo"].index(st.session_state.mode),
)
st.session_state.mode = mode

st.sidebar.markdown("### Files")
st.sidebar.write(f"Data file: `{DATA_FILE}` {'✅' if os.path.exists(DATA_FILE) else '❌'}")
st.sidebar.write(f"Model file: `{MODEL_FILE}` {'✅' if os.path.exists(MODEL_FILE) else '❌'}")

# ------------------ MAIN ------------------
st.title("AI-Based Emotion Detection from Typing Behaviour")
st.subheader("with Music Recommendation System")

st.markdown(
    """
This project captures **typing behaviour** (speed, duration, rhythm) and uses a **Machine Learning model**
to predict your **emotion**. Based on the detected emotion, it recommends **suitable music**.
"""
)

# ---- MODE 1: COLLECT TYPING SAMPLES ----
if mode == "Collect Typing Samples":
    st.header("1️⃣ Collect Typing Data")

    st.info(
        "Step 1: Type the given sentence in the box.\n\n"
        "Step 2: Select your current emotion.\n\n"
        "Step 3: Click **Start Typing**, then after you finish typing click **Submit Sample**."
    )

    sample_sentence = st.text_input(
        "Sentence (you can change if you want):",
        value="Today I am typing to record my mood for the AI project.",
    )

    emotion = st.selectbox("Your current emotion (label for this sample):", ["Happy", "Sad", "Angry", "Calm"])

    text = st.text_area("Type the sentence here:")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Typing Timer"):
            st.session_state.start_time = time.time()
            st.success("Timer started! Now type your text and then click 'Submit Sample'.")
    with col2:
        if st.button("Submit Sample"):
            if st.session_state.start_time is None:
                st.error("Please click 'Start Typing Timer' before submitting.")
            elif not text.strip():
                st.error("Text box is empty. Please type something first.")
            else:
                duration = time.time() - st.session_state.start_time
                features = extract_features(text, duration, emotion)
                save_sample(features)
                st.session_state.start_time = None
                st.success(
                    f"Sample saved! Duration: {duration:.2f} sec | Length: {features['text_length']} chars | Speed: {features['typing_speed_cps']:.2f} chars/sec"
                )

    if os.path.exists(DATA_FILE):
        st.markdown("### Current Dataset Preview")
        df_preview = load_data().tail(10)
        st.dataframe(df_preview)

# ---- MODE 2: TRAIN MODEL ----
elif mode == "Train Model":
    st.header("2️⃣ Train Emotion Detection Model")

    if not os.path.exists(DATA_FILE):
        st.error("No data found. First go to **Collect Typing Samples** and record some data.")
    else:
        df = load_data()
        st.markdown("### Dataset Overview")
        st.write(f"Total samples: {len(df)}")
        st.write(df["emotion"].value_counts())

        if st.button("Train / Retrain Model"):
            with st.spinner("Training model..."):
                result, error = train_model()
            if error:
                st.error(error)
            else:
                st.success(f"Model trained and saved to `{MODEL_FILE}` with accuracy: {result['accuracy']:.2f}")
                st.markdown("#### Classification Report")
                st.text(result["report"])

# ---- MODE 3: EMOTION DETECTION DEMO ----
elif mode == "Emotion Detection Demo":
    st.header("3️⃣ Live Emotion Detection & Music Recommendation")

    model = load_model()
    if model is None:
        st.error("Model file not found. Please go to **Train Model** first.")
    else:
        st.info(
            "Click **Start Typing Timer**, type any text that reflects your current mood, "
            "then click **Detect Emotion**."
        )
        text = st.text_area("Type something about how you feel:")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Typing Timer"):
                st.session_state.start_time = time.time()
                st.success("Timer started. Type your text and then click 'Detect Emotion'.")
        predicted_emotion = None
        with col2:
            if st.button("Detect Emotion"):
                if st.session_state.start_time is None:
                    st.error("Please click 'Start Typing Timer' before detection.")
                elif not text.strip():
                    st.error("Please type something first.")
                else:
                    duration = time.time() - st.session_state.start_time
                    features = extract_features(text, duration)
                    X = pd.DataFrame(
                        [[
                            features["text_length"],
                            features["duration_sec"],
                            features["typing_speed_cps"],
                            features["avg_char_interval"],
                        ]],
                        columns=["text_length", "duration_sec", "typing_speed_cps", "avg_char_interval"],
                    )
                    predicted_emotion = model.predict(X)[0]
                    st.session_state.start_time = None
                    st.success(
                        f"Predicted Emotion: **{predicted_emotion}** "
                        f"(length: {features['text_length']} chars, duration: {features['duration_sec']:.2f} sec)"
                    )

        # MUSIC RECOMMENDATION
        if predicted_emotion:
            st.markdown("---")
            st.subheader("🎵 Recommended Music")
            songs = MUSIC_LIBRARY.get(predicted_emotion, [])
            if not songs:
                st.write("No songs configured for this emotion yet. You can edit `MUSIC_LIBRARY` in the code.")
            else:
                for title, url in songs:
                    st.markdown(f"- [{title}]({url})")

        st.markdown(
            """
            **Note:** You can change the songs and add your own playlists in the `MUSIC_LIBRARY` dictionary
            inside the Python file.
            """
        )
