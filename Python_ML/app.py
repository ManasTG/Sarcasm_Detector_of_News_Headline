# -------------------------------
# IMPORTS
# -------------------------------

import streamlit as st
import matplotlib.pyplot as plt
import pickle
import os
import re

# -------------------------------
# PAGE CONFIG
# -------------------------------

st.set_page_config(
    page_title="Sarcasm Detection Dashboard",
    layout="wide"
)

# -------------------------------
# LOAD MODEL + VECTORIZER
# -------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "svm_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer

model, vectorizer = load_model()

# -------------------------------
# TEXT CLEANING (same as training)
# -------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------

def predict_sarcasm(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])

    result = model.predict(vec)[0]

    # ⚠️ LinearSVC doesn't support predict_proba
    try:
        confidence = model.predict_proba(vec).max()
    except:
        confidence = 0.0

    label = "Sarcastic 😏" if result == 1 else "Not Sarcastic 🙂"

    return label, confidence

# -------------------------------
# SIDEBAR NAVIGATION
# -------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Model Comparison",
        "Model Summary",
        "Project Pros & Cons",
        "Try Model"
    ]
)

# -------------------------------
# PAGE 1 — MODEL COMPARISON
# -------------------------------

if page == "Model Comparison":

    st.title("📊 Model Comparison")

    model_names = ["Naive Bayes", "Logistic Regression", "Random Forest", "SVM"]
    training_times = [1.2, 2.5, 5.1, 3.8]  # replace later

    fig, ax = plt.subplots()

    ax.bar(model_names, training_times)
    ax.set_title("Training Time Comparison")
    ax.set_ylabel("Time (seconds)")
    ax.set_xlabel("Models")

    st.pyplot(fig)

    st.info("Replace training times with real values.")

# -------------------------------
# PAGE 2 — MODEL SUMMARY
# -------------------------------

elif page == "Model Summary":

    st.title("📘 Model Summary")

    st.subheader("Naive Bayes")
    st.write("Fast, simple probabilistic model. Works well on text data.")

    st.subheader("Logistic Regression")
    st.write("Linear model, good balance of speed and accuracy.")

    st.subheader("Random Forest")
    st.write("Ensemble model using multiple decision trees. More accurate but slower.")

    st.subheader("SVM")
    st.write("Powerful classifier for text data. Good performance with proper tuning.")

# -------------------------------
# PAGE 3 — PROS & CONS
# -------------------------------

elif page == "Project Pros & Cons":

    st.title("⚖️ Project Pros & Cons")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("✅ Pros")
        st.write("""
        - Good accuracy
        - Multiple models tested
        - Efficient preprocessing
        - Works on real headlines
        """)

    with col2:
        st.subheader("❌ Cons")
        st.write("""
        - Needs larger dataset
        - Limited sarcasm context understanding
        - Can misclassify complex sentences
        """)

# -------------------------------
# PAGE 4 — TRY MODEL
# -------------------------------

elif page == "Try Model":

    st.title("🧠 Sarcasm Detector")

    st.write("Enter a news headline to check if it's sarcastic.")

    user_input = st.text_area("📰 Enter headline")

    if st.button("Predict"):

        if user_input.strip() == "":
            st.warning("Please enter a headline first.")
        else:
            label, confidence = predict_sarcasm(user_input)

            if "Sarcastic" in label:
                st.error(label)
            else:
                st.success(label)

            if confidence > 0:
                st.write(f"Confidence: {confidence*100:.2f}%")
