# 🎵 AI-Based Emotion Detection & Music Recommendation System

> **BCA Final Year Project** | Python • Streamlit • Machine Learning • NLP

---

## 📌 Overview

An intelligent system that detects a user's **emotional state** by analysing their **typing behaviour** — speed, pauses, errors, keypress duration — and recommends **personalized music** to match or uplift their mood.

Unlike face-based emotion detection, this system works entirely from **how you type**, making it privacy-friendly and usable on any device.

---


## 🧠 How It Works

```
User Types Text
      ↓
Typing Features Extracted
(speed, pauses, errors, duration, patterns)
      ↓
ML Model Predicts Emotion
(Happy / Sad / Angry / Calm)
      ↓
Text Sentiment Analysed (NLP)
      ↓
Music Recommended
(Genre + YouTube + Spotify Links)
```

---

## 🚀 Features

| Feature | Description |
|--------|-------------|
| 🎯 Emotion Detection | Predicts 4 emotions from typing behaviour |
| 🆚 Model Comparison | Trains 4 ML models, picks the best automatically |
| 📊 Analytics Dashboard | Tracks emotion history with interactive charts |
| 📝 Sentiment Analysis | NLP-based text sentiment (Positive/Negative/Neutral) |
| 🎵 Music Recommendation | Genre suggestions + YouTube & Spotify links |
| 🌙 Dark Theme | Professional dark UI built with Streamlit |

---

## 🛠️ Technologies Used

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| ML Models | Scikit-learn (Random Forest, SVM, Gradient Boosting, Logistic Regression) |
| NLP | TextBlob |
| Visualization | Plotly |
| Data | Pandas, NumPy |
| Model Persistence | Joblib |

---

## 📂 Project Structure

```
📁 emotion-music-recommendation/
│
├── app.py                  ← Main Streamlit application
├── train_model.py          ← Model training & comparison script
├── generate_dataset.py     ← Dataset generation script
├── requirements.txt        ← Python dependencies
│
├── emotion_model.pkl       ← Trained ML model (generated)
├── model_info.json         ← Model metadata & accuracy (generated)
├── typing_data.csv         ← Training dataset (generated)
│
└── .streamlit/
    └── config.toml         ← Dark theme configuration
```

---

## ⚙️ How to Run

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Generate dataset & train model
```bash
python generate_dataset.py
python train_model.py
```

### Step 3: Launch the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📊 Model Performance

| Model | Test Accuracy | CV Mean |
|-------|--------------|---------|
| 🏆 Random Forest | ~99% | ~99% |
| SVM | ~99% | ~99% |
| Gradient Boosting | ~99% | ~98% |
| Logistic Regression | ~98% | ~99% |

*Best model is selected automatically during training.*

---

## 🔑 Key Features Analyzed

1. **Typing Speed (WPM)** — Fast = energetic (Happy/Angry), Slow = calm/sad
2. **Keypress Duration (ms)** — Longer hold = sad/calm, Short = happy/angry
3. **Pause Between Words (ms)** — Long pauses = sad/calm
4. **Error Rate** — High errors = angry/excited
5. **Backspace Count** — More deletions = agitated or uncertain
6. **Exclamation Frequency** — High = happy or angry
7. **Sentence Length** — Short = sad/angry, Long = calm/happy
8. **Avg Word Length** — Longer words = calm, thoughtful

---

## ⚠️ Limitations

- Typing speed is entered manually (real-time capture needs JavaScript integration)
- Dataset is simulated based on research patterns, not collected from real users
- 4 emotion classes only (Happy, Sad, Angry, Calm)
- Music recommendations are category-based, not truly personalized

---

## 🔮 Future Improvements

- [ ] Real-time keystroke capture via JavaScript + Streamlit component
- [ ] Facial emotion detection using OpenCV + DeepFace
- [ ] Spotify API for actual song-level recommendations
- [ ] User feedback loop for continuous model improvement
- [ ] Mobile-friendly PWA version
- [ ] Collaborative filtering for personalized music

---

## 👨‍💻 Developer

**BCA Final Year Project** | 2023-26 
Department of Computer Applications
https://github.com/mitalidas125/AI-based-emotion-detection-music-recommendation/edit/main/README.md

---

## 📚 References

1. Khanna, P., & Sasikumar, M. (2010). Recognising Emotions from Keyboard Stroke Pattern. *IJCA*
2. Epp, C., Lippold, M., & Mandryk, R. L. (2011). Identifying Emotional States Using Keystroke Dynamics. *CHI*
3. Breiman, L. (2001). Random Forests. *Machine Learning*
4. Bollen, J., Mao, H., & Zeng, X. (2011). Twitter Mood Predicts the Stock Market. *Journal of Computational Science*
