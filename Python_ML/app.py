import streamlit as st
import matplotlib.pyplot as plt

# -------------------------------
# Page Config
# -------------------------------

st.set_page_config(
    page_title="Sarcasm Detection Dashboard",
    layout="wide"
)

# -------------------------------
# Sidebar Navigation
# -------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Model Comparison",
        "Model Summary",
        "Project Pros & Cons"
    ]
)

# -------------------------------
# PAGE 1 — MODEL COMPARISON
# -------------------------------

if page == "Model Comparison":

    st.title("Model Comparison")
    st.subheader("Training Time Comparison (Bar Graph)")

    # Placeholder Data (Replace Later)

    model_names = [
        "Naive Bayes",
        "Logistic Regression",
        "Random Forest",
        "SVM"
    ]

    training_times = [
        1.2,   # Replace later
        2.5,
        5.1,
        3.8
    ]

    # Create Bar Graph

    fig, ax = plt.subplots()

    ax.bar(
        model_names,
        training_times
    )

    ax.set_title("Model Training Time")
    ax.set_ylabel("Time (seconds)")
    ax.set_xlabel("Models")

    st.pyplot(fig)

    st.info(
        "Replace the training_times list with actual values later."
    )

# -------------------------------
# PAGE 2 — MODEL SUMMARY
# -------------------------------

elif page == "Model Summary":

    st.title("Model Summary")

    st.subheader("Naive Bayes")

    st.write("""
    Replace this text with:
    - Model description
    - Accuracy
    - Training time
    - Key observations
    """)

    st.subheader("Logistic Regression")

    st.write("""
    Replace this text with:
    - Model description
    - Accuracy
    - Training time
    - Key observations
    """)

    st.subheader("Random Forest")

    st.write("""
    Replace this text with:
    - Model description
    - Accuracy
    - Training time
    - Key observations
    """)

    st.subheader("SVM")

    st.write("""
    Replace this text with:
    - Model description
    - Accuracy
    - Training time
    - Key observations
    """)

# -------------------------------
# PAGE 3 — PROS & CONS
# -------------------------------

elif page == "Project Pros & Cons":

    st.title("Project Pros & Cons")

    col1, col2 = st.columns(2)

    # Pros Column

    with col1:

        st.subheader("Pros")

        st.write("""
        - Add your project strengths here
        - Example:
            - High accuracy
            - Multiple models tested
            - Efficient preprocessing
        """)

    # Cons Column

    with col2:

        st.subheader("Cons")

        st.write("""
        - Add project limitations here
        - Example:
            - Needs larger dataset
            - Limited sarcasm context detection
            - Requires more hyperparameter tuning
        """)

    st.warning(
        "Edit these lists based on your real project findings."
    )




    # Page 1 — Model Comparison Graph
    import streamlit as st
import pandas as pd

st.title("📊 Model Comparison (Execution Time)")

st.write("Comparison of 4 models based on execution time.")

# Placeholder data — EDIT LATER
data = {
    "Model": ["SVM", "Naive Bayes", "Logistic Regression", "Random Forest"],
    "Time": [0.0, 0.0, 0.0, 0.0]  # Replace later
}

df = pd.DataFrame(data)

st.bar_chart(
    data=df,
    x="Model",
    y="Time"
)

st.info("Update time values later.")


# Page 2 — Model Summary


import streamlit as st

st.title("📘 Model Summary")

st.write("Summary of all 4 machine learning models.")

# Placeholder summaries

st.subheader("1️⃣ SVM Model")

st.write("""
Write summary of SVM model here.

Example:
- Kernel used
- Accuracy
- Training method
""")

st.subheader("2️⃣ Naive Bayes")

st.write("""
Write summary of Naive Bayes here.
""")

st.subheader("3️⃣ Logistic Regression")

st.write("""
Write summary here.
""")

st.subheader("4️⃣ Random Forest")

st.write("""
Write summary here.
""")


# Page 3 — Pros and Cons
import streamlit as st

st.title("⚖️ Pros and Cons of Our Project")

# Pros Section

st.subheader("✅ Pros")

st.write("""
- Fast prediction  
- Easy to use  
- Lightweight model  
- Good accuracy  
- Works on real headlines  

(Add your own later)
""")

# Cons Section

st.subheader("❌ Cons")

st.write("""
- Limited dataset  
- May misclassify rare sarcasm  
- Depends on training quality  

(Add your own later)
""")