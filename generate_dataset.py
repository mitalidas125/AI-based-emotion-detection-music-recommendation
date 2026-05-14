"""
Dataset Generator for Emotion Detection
Generates realistic typing behaviour data for 4 emotions
"""

import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

# Realistic typing patterns per emotion (based on psychology research)
# Happy: moderate-fast typing, less errors, smooth
# Sad: slow typing, more pauses, longer key hold
# Angry: fast aggressive typing, more errors, hard keypresses
# Calm: steady moderate typing, minimal errors

def generate_samples(emotion, n=300):
    samples = []

    for _ in range(n):
        if emotion == "Happy":
            typing_speed     = np.random.normal(65, 8)      # WPM
            keypress_duration= np.random.normal(80, 10)     # ms
            pause_duration   = np.random.normal(150, 30)    # ms between words
            error_rate       = np.random.normal(0.03, 0.01) # fraction
            backspace_count  = np.random.normal(2, 1)
            avg_word_length  = np.random.normal(5.2, 0.5)
            sentence_length  = np.random.normal(12, 2)
            exclamation_use  = np.random.normal(0.6, 0.2)   # frequency

        elif emotion == "Sad":
            typing_speed     = np.random.normal(38, 7)
            keypress_duration= np.random.normal(130, 20)
            pause_duration   = np.random.normal(400, 80)
            error_rate       = np.random.normal(0.05, 0.02)
            backspace_count  = np.random.normal(4, 1.5)
            avg_word_length  = np.random.normal(4.5, 0.6)
            sentence_length  = np.random.normal(7, 2)
            exclamation_use  = np.random.normal(0.05, 0.03)

        elif emotion == "Angry":
            typing_speed     = np.random.normal(85, 12)
            keypress_duration= np.random.normal(60, 12)
            pause_duration   = np.random.normal(80, 20)
            error_rate       = np.random.normal(0.10, 0.03)
            backspace_count  = np.random.normal(7, 2)
            avg_word_length  = np.random.normal(4.8, 0.7)
            sentence_length  = np.random.normal(9, 3)
            exclamation_use  = np.random.normal(0.8, 0.15)

        elif emotion == "Calm":
            typing_speed     = np.random.normal(55, 6)
            keypress_duration= np.random.normal(95, 10)
            pause_duration   = np.random.normal(200, 40)
            error_rate       = np.random.normal(0.02, 0.01)
            backspace_count  = np.random.normal(1.5, 0.8)
            avg_word_length  = np.random.normal(5.5, 0.4)
            sentence_length  = np.random.normal(14, 2)
            exclamation_use  = np.random.normal(0.1, 0.05)

        # Clip to realistic values
        typing_speed     = max(10, min(150, typing_speed))
        keypress_duration= max(30, min(300, keypress_duration))
        pause_duration   = max(50, min(1000, pause_duration))
        error_rate       = max(0, min(0.3, error_rate))
        backspace_count  = max(0, min(20, backspace_count))
        avg_word_length  = max(2, min(10, avg_word_length))
        sentence_length  = max(3, min(25, sentence_length))
        exclamation_use  = max(0, min(1, exclamation_use))

        samples.append({
            "typing_speed_wpm"    : round(typing_speed, 2),
            "keypress_duration_ms": round(keypress_duration, 2),
            "pause_duration_ms"   : round(pause_duration, 2),
            "error_rate"          : round(error_rate, 4),
            "backspace_count"     : round(backspace_count, 1),
            "avg_word_length"     : round(avg_word_length, 2),
            "sentence_length"     : round(sentence_length, 1),
            "exclamation_freq"    : round(exclamation_use, 3),
            "emotion"             : emotion
        })

    return samples

# Generate dataset
all_data = []
for emotion in ["Happy", "Sad", "Angry", "Calm"]:
    all_data.extend(generate_samples(emotion, n=300))

df = pd.DataFrame(all_data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
df.to_csv("typing_data.csv", index=False)

print(f"✅ Dataset generated: {len(df)} samples")
print(df["emotion"].value_counts())
print("\nSample data:")
print(df.head(8))