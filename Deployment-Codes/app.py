# import streamlit as st
# import torch
# from transformers import AutoTokenizer
# from model import EmotionClassifier  # model.py same folder me hona chahiye
# from huggingface_hub import hf_hub_download

# # ---------------- CONFIG ----------------
# HF_REPO_ID = "rajanshu22f3/best_model_deberta"
# MODEL_FILENAME = "best_model_deberta.pt"
# MODEL_NAME = "microsoft/deberta-v3-large"
# EMOTIONS = ["anger", "fear", "joy", "sadness", "surprise"]
# MAX_LEN = 128

# st.set_page_config(page_title="Emotion Classifier", page_icon="😊")

# st.title("🧠 DeBERTa Emotion Classifier")
# st.write("This app classifies emotions from text using your custom trained model on Hugging Face Hub.")

# # ---------------- DOWNLOAD MODEL FROM HF ----------------
# @st.cache_resource
# def load_model_and_tokenizer():
#     # Download model from Hugging Face
#     model_path = hf_hub_download(
#         repo_id=HF_REPO_ID,
#         filename=MODEL_FILENAME
#     )
#     # Load tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
#     # Load model architecture
#     model = EmotionClassifier(model_name=MODEL_NAME, num_labels=len(EMOTIONS))
#     # Load trained weights
#     state = torch.load(model_path, map_location=torch.device("cpu"))
#     model.load_state_dict(state)
#     model.eval()
#     return tokenizer, model

# with st.spinner("Loading model from Hugging Face..."):
#     tokenizer, model = load_model_and_tokenizer()
# st.success("✅ Model loaded successfully!")

# # ---------------- TEXT CLEANING ----------------
# def clean_text(text):
#     import re, emoji
#     from ftfy import fix_text
#     text = fix_text(text)
#     text = re.sub(r'http\S+|www.\S+', '<URL>', text)
#     text = re.sub(r'@\w+', '<USER>', text)
#     text = emoji.demojize(text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# # ---------------- USER INPUT ----------------
# user_text = st.text_area("Enter text to classify:", "I am really happy today!")

# if st.button("Classify"):
#     if user_text.strip() == "":
#         st.warning("Please enter some text.")
#     else:
#         cleaned_text = clean_text(user_text)
#         inputs = tokenizer(
#             cleaned_text,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=MAX_LEN
#         )
#         with torch.no_grad():
#             logits = model(inputs["input_ids"], inputs["attention_mask"])
#             probs = torch.softmax(logits, dim=-1).squeeze()
#             predicted_idx = torch.argmax(probs).item()
#             predicted_emotion = EMOTIONS[predicted_idx]

#         st.metric(label="Predicted Emotion", value=predicted_emotion)
#         with st.expander("View Probabilities"):
#             for emo, p in zip(EMOTIONS, probs.tolist()):
#                 st.write(f"{emo}: {p:.4f}")



import streamlit as st
import torch
from transformers import AutoTokenizer
from model import EmotionClassifier  # model.py same folder me hona chahiye
from huggingface_hub import hf_hub_download

# ---------------- CONFIG ----------------
HF_REPO_ID = "rajanshu22f3/best_model_deberta"
MODEL_FILENAME = "best_model_deberta.pt"
MODEL_NAME = "microsoft/deberta-v3-large"
EMOTIONS = ["anger", "fear", "joy", "sadness", "surprise"]
MAX_LEN = 128

st.set_page_config(page_title="Emotion Classifier", page_icon="😊")

st.title("🧠 DeBERTa Emotion Classifier")
st.write("This app classifies emotions from text using your custom trained model on Hugging Face Hub.")


# ---------------- SAFE MODEL + TOKEN LOADING ----------------
@st.cache_resource
def load_model_and_tokenizer():
    try:
        st.write("📥 Downloading model from Hugging Face Hub...")

        # Download .pt file
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME
        )

        st.write("🔧 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

        st.write("🧠 Initializing model architecture...")
        model = EmotionClassifier(model_name=MODEL_NAME, num_labels=len(EMOTIONS))

        st.write("📌 Loading fine-tuned weights...")
        state = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state)
        model.eval()

        st.success("✅ Model & Tokenizer Loaded Successfully!")
        return tokenizer, model

    except Exception as e:
        st.error("❌ ERROR: Failed to load model or tokenizer.")
        st.error(f"Error Details:\n\n{str(e)}")
        return None, None


# ---------------- LOAD EVERYTHING ----------------
with st.spinner("Loading model from Hugging Face..."):
    tokenizer, model = load_model_and_tokenizer()

# Stop app if loading failed
if tokenizer is None or model is None:
    st.stop()


# ---------------- TEXT CLEANING FUNCTION ----------------
def clean_text(text):
    import re, emoji
    from ftfy import fix_text

    text = fix_text(text)
    text = re.sub(r'http\S+|www.\S+', '<URL>', text)
    text = re.sub(r'@\w+', '<USER>', text)
    text = emoji.demojize(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------- USER INPUT ----------------
user_text = st.text_area("Enter text to classify:", "I am really happy today!")

if st.button("Classify"):
    try:
        if user_text.strip() == "":
            st.warning("Please enter some text.")
        else:
            cleaned = clean_text(user_text)

            inputs = tokenizer(
                cleaned,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN
            )

            with torch.no_grad():
                logits = model(inputs["input_ids"], inputs["attention_mask"])
                probs = torch.softmax(logits, dim=-1).squeeze()
                idx = torch.argmax(probs).item()
                predicted = EMOTIONS[idx]

            st.metric(label="Predicted Emotion", value=predicted)

            with st.expander("View Probabilities"):
                for emo, p in zip(EMOTIONS, probs.tolist()):
                    st.write(f"{emo}: {p:.4f}")

    except Exception as e:
        st.error("❌ Classification failed.")
        st.error(str(e))
