import streamlit as st
import torch
import os
from transformers import AutoTokenizer
from model import EmotionClassifier

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model_deberta.pt")
MODEL_NAME = "microsoft/deberta-v3-large"
EMOTIONS = ["anger", "fear", "joy", "sadness", "surprise"]
MAX_LEN = 128

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    import re, emoji
    from ftfy import fix_text
    text = fix_text(text)
    text = re.sub(r'http\S+|www.\S+', '<URL>', text)
    text = re.sub(r'@\w+', '<USER>', text)
    text = emoji.demojize(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------- STREAMLIT ----------------
st.set_page_config(page_title="Emotion Classifier", page_icon="🎭")
st.title("🎭 Emotion Classification with DeBERTa")
st.write("This app uses your **locally saved fine-tuned DeBERTa model**.")

# Model File Check
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found at path:\n`{MODEL_PATH}`")
    st.stop()

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    try:
        st.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        st.info("Loading model architecture...")
        model = EmotionClassifier(model_name=MODEL_NAME, num_labels=5)

        st.info("Loading trained weights (.pt)...")
        state = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state)

        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

with st.spinner("Initializing model..."):
    tokenizer, model = load_model()

if tokenizer is None or model is None:
    st.stop()

st.success("✅ Model loaded successfully!")

# ---------------- INPUT AREA ----------------
text = st.text_area(
    "Enter text to analyze emotion:",
    "God knows what my brother will bring for me!"
)

if st.button("Predict Emotion"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(text)

        # Tokenize
        inputs = tokenizer(
            cleaned,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN
        )

        with torch.no_grad():
            logits = model(inputs["input_ids"], inputs["attention_mask"])
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()

        # Highest Emotion
        predicted_idx = probs.index(max(probs))
        predicted_emotion = EMOTIONS[predicted_idx]

        # Show Prediction
        st.metric("Predicted Emotion", predicted_emotion.upper())

        # Show Probabilities
        st.subheader("🔢 Emotion Probabilities")
        for emo, p in zip(EMOTIONS, probs):
            st.write(f"**{emo.capitalize()}**: `{p:.4f}`")

        # Optional bar chart
        st.bar_chart(
            {emo: p for emo, p in zip(EMOTIONS, probs)}
        )
