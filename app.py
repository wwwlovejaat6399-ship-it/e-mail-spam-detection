import streamlit as st
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

st.set_page_config(
    page_title="Spam Detection Dashboard",
    page_icon="📩",
    layout="wide"
)

st.title("📩 Email & SMS Spam Detection")
st.write("Machine Learning Performance Dashboard")

# Load trained model
with open("spam_model.pkl", "rb") as file:
    model = pickle.load(file)

# Dashboard
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose Page",
    ["Spam Checker", "Model Dashboard"]
)

# ---------------- SPAM CHECKER ----------------
if page == "Spam Checker":

    st.header("Check Your Message")

    message = st.text_area("Enter your email or SMS")

    if st.button("Predict Spam"):
        if message.strip():

            prediction = model.predict([message])[0]
            probability = model.predict_proba([message])[0]

            classes = list(model.classes_)
            spam_index = classes.index("spam")
            spam_probability = probability[spam_index] * 100

            if prediction == "spam":
                st.error("⚠️ Spam Message")
            else:
                st.success("✅ Not Spam")

            st.metric(
                "Estimated Spam Probability",
                f"{spam_probability:.2f}%"
            )

        else:
            st.warning("Please enter a message.")

# ---------------- MODEL DASHBOARD ----------------
elif page == "Model Dashboard":

    st.header("📊 Model Performance Dashboard")

    try:
        # Load dataset
        data = pd.read_csv(
            "SMSSpamCollection",
            sep="\t",
            names=["label", "message"]
        )

        X = data["message"]
        y = data["label"]

        # Same split used during training
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Predict test messages
        predictions = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, pos_label="spam",
            zero_division=0
        )
        recall = recall_score(
            y_test, predictions, pos_label="spam",
            zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, pos_label="spam",
            zero_division=0
        )

        # Metric cards
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        col1.metric("Accuracy", f"{accuracy * 100:.2f}%")
        col2.metric("Precision", f"{precision * 100:.2f}%")
        col3.metric("Recall", f"{recall * 100:.2f}%")
        col4.metric("F1 Score", f"{f1 * 100:.2f}%")

        # Confusion matrix
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=["ham", "spam"]
        )

        cm_df = pd.DataFrame(
            cm,
            index=["Actual Not Spam", "Actual Spam"],
            columns=["Predicted Not Spam", "Predicted Spam"]
        )

        st.dataframe(cm_df, use_container_width=True)

        st.info(
            "Metrics are calculated on the held-out test dataset."
        )

    except FileNotFoundError:
        st.error(
            "SMSSpamCollection file nahi mili. "
            "Dataset ko project folder mein rakho."
        )
    except Exception as e:
        st.error(f"Error: {e}")