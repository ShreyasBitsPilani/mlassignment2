import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="ML Assignment 2 – Classification Models",
    layout="wide"
)

st.title("Machine Learning Classification Models")
st.write("BITS Assignment 2 – Model Comparison using Streamlit")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("User Inputs")

model_name = st.sidebar.selectbox(
    "Select Classification Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbors",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    )
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

# -------------------------------
# Helper Functions
# -------------------------------
def get_model(model_name):
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=1000)
    elif model_name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42)
    elif model_name == "K-Nearest Neighbors":
        return KNeighborsClassifier(n_neighbors=5)
    elif model_name == "Naive Bayes":
        return GaussianNB()
    elif model_name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        return XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        )

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Classification Report": classification_report(y_test, y_pred)
    }
    return metrics

# -------------------------------
# Main App Logic
# -------------------------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(data.head())

    if "target" not in data.columns:
        st.error("The uploaded CSV must contain a column named 'target'")
    else:
        X = data.drop("target", axis=1)
        y = data["target"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = get_model(model_name)
        model.fit(X_scaled, y)

        results = evaluate_model(model, X_scaled, y)

        st.subheader("Evaluation Metrics")

        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"],
            "Value": [
                results["Accuracy"],
                results["AUC"],
                results["Precision"],
                results["Recall"],
                results["F1 Score"],
                results["MCC"]
            ]
        })

        st.table(metrics_df)

        st.subheader("Confusion Matrix")
        st.write(results["Confusion Matrix"])

        st.subheader("Classification Report")
        st.text(results["Classification Report"])

else:
    st.info("Please upload a CSV file containing test data with a 'target' column.")
