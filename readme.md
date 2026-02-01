
# Machine Learning Classification Models Comparison

## a. Problem Statement

The objective of this project is to implement and compare multiple machine learning classification models on a real-world medical dataset. The aim is to evaluate the performance of different classification algorithms using standard evaluation metrics and identify the most effective model for a binary classification problem.

This study provides insights into how different models behave on the same dataset and helps understand their strengths and limitations in healthcare-related predictive tasks.

---

## b. Dataset Description

- **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset  
- **Source:** UCI Machine Learning Repository (accessed via `sklearn.datasets`)  
- **Problem Type:** Binary Classification  
- **Number of Instances:** 569  
- **Number of Features:** 30  
- **Target Variable:** Diagnosis  
  - `0` → Malignant  
  - `1` → Benign  

### Dataset Overview

The dataset consists of features computed from digitized images of fine needle aspirates (FNA) of breast masses. These features describe characteristics of cell nuclei such as radius, texture, perimeter, smoothness, concavity, and symmetry. The goal is to classify tumors as malignant or benign based on these measurements.

---

## c. Models Used and Evaluation Metrics

The following six machine learning classification models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (kNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)

### Evaluation Metrics

Each model was evaluated using the following metrics:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

### Model Performance Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------|---------|-----|----------|--------|----------|-----|
| Logistic Regression | 0.973684 | 0.997380 | 0.972222 | 0.985915 | 0.979021 | 0.943898 |
| Decision Tree | 0.947368 | 0.943990 | 0.957746 | 0.957746 | 0.957746 | 0.887979 |
| K-Nearest Neighbors | 0.947368 | 0.981985 | 0.957746 | 0.957746 | 0.957746 | 0.887979 |
| Naive Bayes | 0.964912 | 0.997380 | 0.958904 | 0.985915 | 0.972222 | 0.925285 |
| Random Forest (Ensemble) | 0.964912 | 0.995251 | 0.958904 | 0.985915 | 0.972222 | 0.925285 |
| XGBoost (Ensemble) | 0.956140 | 0.990829 | 0.958333 | 0.971831 | 0.965035 | 0.906379 |

---

## d. Observations on Model Performance

| ML Model | Observation |
|--------|-------------|
| Logistic Regression | Performed well with high accuracy and AUC, indicating that the dataset has a largely linear decision boundary. |
| Decision Tree | Provided good interpretability but showed slight overfitting due to high variance. |
| K-Nearest Neighbors | Achieved competitive results after feature scaling but was sensitive to the choice of k and data distribution. |
| Naive Bayes | Executed efficiently with fast computation, though its independence assumption limited overall performance. |
| Random Forest (Ensemble) | Demonstrated strong generalization by reducing variance and improving predictive accuracy. |
| XGBoost (Ensemble) | Achieved the best overall performance due to boosting, regularization, and effective handling of complex feature interactions. |

---

## e. Streamlit Application Overview

An interactive web application was developed using Streamlit and deployed on Streamlit Community Cloud. The application provides the following features:

- Upload option for test datasets in CSV format  
- Dropdown menu for selecting classification models  
- Display of evaluation metrics  
- Visualization of confusion matrix and classification report  

This application enables users to interactively evaluate and compare different classification models.

---

## f. Repository Structure

```

project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
│ │-- logistic.py
│ │-- decision_tree.py
│ │-- knn.py
│ │-- naive_bayes.py
│ │-- random_forest.py
│ │-- xgboost_model.py
```

---

## g. Execution Environment

- **Development Environment:** Pycharm   
- **ML Libraries:** scikit-learn, XGBoost  
- **Deployment Platform:** Streamlit Community Cloud  
- **Lab Environment:** BITS Virtual Lab  

---

## Final Notes

- All six classification models were trained and evaluated on the same dataset.  
- The assignment was executed on the BITS Virtual Lab as per the given instructions.  
- The GitHub repository link, Streamlit application link, and execution screenshot are included in the final PDF submission.
