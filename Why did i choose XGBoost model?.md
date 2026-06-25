# 🤖 Why XGBoost?

> *Selecting the right model is just as important as building one.*

This document explains why **XGBoost** was chosen as the final model for this project and provides additional resources covering the complete exploratory data analysis and model development pipeline.

---

# 🎯 Project Objective

The objective of this project was to build a robust machine learning model capable of detecting **fraudulent credit card transactions** from a **highly imbalanced dataset** while maximizing fraud detection performance.

The dataset contains:

| Property                |                 Value |
| ----------------------- | --------------------: |
| Total Transactions      |           **284,807** |
| Fraudulent Transactions |               **492** |
| Fraud Percentage        |             **0.17%** |
| Problem Type            | Binary Classification |

Because fraudulent transactions represent only a tiny fraction of the data, overall **accuracy is not an appropriate evaluation metric**.

Instead, the models were evaluated using:

* Precision
* Recall
* F1-Score
* ROC-AUC
* Precision-Recall (PR) Curve

---

# ⚖️ Models Evaluated

The following ensemble boosting algorithms were trained and compared:

| Model                        |    Status   |
| :--------------------------- | :---------: |
| AdaBoost Classifier          | ✅ Evaluated |
| Gradient Boosting Classifier | ✅ Evaluated |
| **XGBoost Classifier**       | 🏆 Selected |

After evaluating all three models, **XGBoost consistently achieved the strongest overall performance**, providing the best balance between Precision, Recall, F1-Score, and ROC-AUC.

To further optimize performance, the selected model was tuned using **RandomizedSearchCV** before being serialized as **`model.pkl`** for inference.

---

# 🏆 Why XGBoost?

XGBoost was selected because it demonstrated superior predictive performance compared to the other boosting algorithms evaluated.

Some of the key advantages observed during experimentation were:

* Better balance between Precision and Recall.
* Higher F1-Score.
* Strong ROC-AUC performance.
* Better generalization on unseen transactions.
* Efficient handling of complex decision boundaries.

These characteristics made XGBoost the most suitable model for this fraud detection task.

---

# 🔒 Dataset Confidentiality

> [!IMPORTANT]
> The original transaction attributes are **not publicly available**.

To preserve customer privacy, the dataset providers anonymized the original transaction information by applying **Principal Component Analysis (PCA)** before releasing the dataset.

Consequently:

* **V1–V28** are engineered PCA components.
* **Time** and **Amount** are the only original features.
* The original transaction attributes cannot be reconstructed.

Although the trained model can technically be served through an API, these engineered features do not correspond to meaningful transaction information that an end user can provide. Therefore, this project was developed primarily as a **learning project** to understand the complete machine learning workflow, model evaluation, and backend integration using **FastAPI**.

---

# 📚 Additional Resources

## 📈 Exploratory Data Analysis (EDA)

A detailed notebook covering:

* Data Understanding
* Feature Analysis
* Class Distribution
* Visualizations
* Insights

➡️ **[View the Complete EDA Notebook on Kaggle](https://www.kaggle.com/code/ashuthetics007/credit-card-fraud-detection-dataset-eda)**

---

## 🤖 XGBoost Model Development Pipeline

Complete notebook containing:

* Data Preparation
* Model Comparison
* Hyperparameter Tuning
* Model Evaluation
* Final XGBoost Pipeline
* Model Serialization (`model.pkl`)

➡️ **[View the Complete XGBoost Pipeline Notebook on Kaggle](https://www.kaggle.com/code/ashuthetics007/xgb-model)**

---

# 💻 Source Code

The complete source code for this project is available on GitHub.

➡️ **[Credit Card Fraud Detection Repository](https://github.com/Ashushukla007/Credit_Card_Fraud_Detection_)**

---

# 🛠️ Tech Stack

| Category              | Technologies          |
| --------------------- | --------------------- |
| Programming Language  | Python                |
| Data Analysis         | Pandas, NumPy         |
| Visualization         | Matplotlib, Seaborn   |
| Machine Learning      | Scikit-learn, XGBoost |
| Hyperparameter Tuning | RandomizedSearchCV    |
| Backend API           | FastAPI               |

---

## ⭐ Thank you for taking the time to explore this project!

If you found this repository helpful or interesting, consider giving it a ⭐ on GitHub.
