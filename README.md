# 🛡️ Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python) ![FastAPI](https://img.shields.io/badge/FastAPI-0.138+-009688?style=flat-square&logo=fastapi) ![Streamlit](https://img.shields.io/badge/Streamlit-Latest-FF4B4B?style=flat-square&logo=streamlit) ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=flat-square&logo=scikitlearn) ![Pickle](https://img.shields.io/badge/Model-Pickle-FFD700?style=flat-square) ![License](https://img.shields.io/badge/License-MIT-green?style=flat-square) ![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen?style=flat-square) ![Research](https://img.shields.io/badge/Purpose-Research--Demo-blue?style=flat-square)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [⚠️ Critical Dataset Disclaimer](#️-critical-dataset-disclaimer)
- [Why the Dataset Cannot Be Understood by End Users](#why-the-dataset-cannot-be-understood-by-end-users)
- [Access to Sample Data](#access-to-sample-data)
- [Model Information](#model-information)
- [Project Architecture](#project-architecture)
- [Features](#features)
- [Installation & Setup](#installation--setup)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Running the Application Locally](#running-the-application-locally)
- [Example Request & Response](#example-request--response)
- [Deployment Possibilities](#deployment-possibilities)
- [Future Improvements](#future-improvements)
- [Limitations](#limitations)
- [Disclaimer](#disclaimer)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## 🎯 Project Overview

**Credit Card Fraud Detection System** is a full-stack machine learning application that classifies credit card transactions as fraudulent or legitimate. The project demonstrates:

- ✅ **End-to-End ML Pipeline**: Data preprocessing, model training, and serialization
- ✅ **Production-Ready Deployment**: FastAPI backend with async request handling
- ✅ **Interactive User Interface**: Streamlit frontend for real-time predictions
- ✅ **API-Driven Architecture**: RESTful endpoints with Pydantic validation
- ✅ **Research-Grade Documentation**: Transparent about data limitations and model behavior

### Dataset Details

| Metric | Value |
|--------|-------|
| **Source** | [ULB Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **Total Transactions** | 284,807 |
| **Fraudulent Cases** | 492 (0.17%) |
| **Legitimate Cases** | 284,315 (99.83%) |
| **Features** | 30 (Time, Amount, V1-V28 [PCA-transformed]) |
| **Dataset Size** | ~150 MB |
| **Class Imbalance Ratio** | 1:578 (Fraud to Legitimate) |

---

## ⚠️ CRITICAL DATASET DISCLAIMER

### 🔴 IMPORTANT: Features Are Anonymized and PCA-Transformed

The dataset published on Kaggle has **already undergone extensive preprocessing** by the dataset creators before public release. Specifically:

1. **Original banking features were transformed using Principal Component Analysis (PCA)**
2. **Features V1–V28 are NOT the original transaction attributes**
3. **The original features have been intentionally anonymized for privacy and security**
4. **No mappings exist from V1–V28 back to real transaction properties**

### What This Means for This Project

This project **demonstrates machine learning model development and deployment** but **is NOT an explainable fraud detection system** because:

- 🔴 Features have no semantic meaning
- 🔴 Feature importance cannot be interpreted as business insights
- 🔴 Predictions cannot be explained in domain-specific terms
- 🔴 The model identifies mathematical patterns but not business reasons for fraud

---

## Why the Dataset Cannot Be Understood by End Users

### 📊 PCA Transformation Explained

**Principal Component Analysis (PCA)** is a dimensionality reduction technique that:

1. **Takes original features** (e.g., merchant type, location, purchase history, etc.)
2. **Converts them into uncorrelated mathematical components** (V1–V28)
3. **Preserves statistical information** while removing semantic meaning
4. **Makes the data uninterpretable** at the individual feature level

### Why Original Features Were Hidden

**Banking Privacy & Security Reasons:**
- ✅ Protects customer confidentiality
- ✅ Prevents reverse-engineering of fraud patterns
- ✅ Complies with data protection regulations (GDPR, PCI-DSS)
- ✅ Prevents adversarial attacks on the financial system

### Consequences for Model Interpretability

| Aspect | Challenge |
|--------|-----------|
| **Feature Meaning** | V1–V28 have no human-readable interpretation |
| **Business Logic** | Cannot explain why a transaction is flagged as fraud |
| **Regulatory Compliance** | Cannot provide audit trails based on original variables |
| **End-User Trust** | Users cannot understand model decisions |
| **Pattern Discovery** | Statistical patterns exist but lack business context |

### Example: What We CAN'T Know

```
V1 = -1.3598 → Could represent ANY combination of original features
V5 = 0.2310  → Unknown banking metrics compressed into one number
V14 = 1.0457 → No way to link this to actual transaction properties

❌ We CANNOT say:
   "Fraud was detected because the merchant type was X"
   "High-risk because the purchase location was Y"
   "Anomaly detected in customer's spending pattern Z"

✅ We CAN say:
   "Statistical anomaly detected in feature space"
   "Pattern matches historical fraudulent transaction signature"
```

### Recommendation

**This project is suitable for:**
- Educational purposes and demonstrating ML engineering
- Understanding fraud detection model architecture
- Learning how to build production ML systems
- Portfolio demonstration of full-stack ML skills

**This project is NOT suitable for:**
- Real-world banking fraud detection (without original features)
- Making actual financial decisions
- Regulatory reporting (without explainability)
- Business intelligence insights

---

## Access to Sample Data

### What Sample Data Is Provided

A subset of **982 transformed rows** is included in this repository for demonstration and testing purposes.

### Important Characteristics of Sample Data

| Property | Details |
|----------|---------|
| **Row Count** | 982 transactions |
| **Purpose** | API input validation, testing, demonstrations |
| **Format** | PCA-transformed features (identical to training data) |
| **Interpretability** | ❌ NOT human-readable or explainable |
| **Representativeness** | Reflects the class imbalance of full dataset |
| **Use Case** | Testing prediction endpoints, UI demonstrations |

### How to Use Sample Data

```python
# Example: Understanding the data structure
sample_transaction = {
    "time": 0,
    "amount": 149.62,
    "v1": -1.3598071336738,
    "v2": -0.0727236390324,
    "v3": 2.36060841498296,
    ...
    "v28": 0.271450160456249
}

# This transaction contains ONLY anonymized PCA-transformed values
# You can send it to the API and get a fraud probability
# But you cannot understand WHY the prediction was made
```

### What Sample Data DOES NOT Represent

- ❌ Raw banking transaction records
- ❌ Original customer information
- ❌ Real-world interpretable features
- ❌ Explainable decision factors
- ❌ Production-ready fraud detection data

---

## Model Information

### Problem Type: Binary Classification

**Objective:** Classify each transaction as:
- **Class 0:** Legitimate Transaction ✅
- **Class 1:** Fraudulent Transaction ⚠️

### Class Imbalance: A Critical Challenge

**Data Distribution:**

```
Legitimate Transactions:  284,315 (99.83%) ███████████████████████████████████████ 
Fraudulent Transactions:     492 (0.17%)  █
```

**Imbalance Ratio: 1:578**

### Why Imbalance Requires Special Handling

#### Problem with Standard Metrics

```
Naive Baseline Model: "Always predict Legitimate"
├─ Accuracy: 99.83% ✅ (looks great!)
├─ Precision: N/A (never predicts fraud)
├─ Recall: 0% ❌ (catches 0 fraud cases)
└─ F1-Score: 0% ❌ (completely useless)

❌ A model that always says "legitimate" achieves 99.83% accuracy!
❌ Accuracy is MISLEADING for imbalanced datasets
```

#### Proper Evaluation Metrics for Fraud Detection

| Metric | Formula | Why It Matters |
|--------|---------|----------------|
| **Precision** | TP/(TP+FP) | Of all fraud alerts, how many are correct? (Cost of false alarms) |
| **Recall** | TP/(TP+FN) | Of all actual fraud, how many did we catch? (Cost of missed fraud) |
| **F1-Score** | 2×(P×R)/(P+R) | Balance between precision and recall |
| **ROC-AUC** | Area under ROC curve | Overall model discrimination ability |
| **PR-AUC** | Precision-Recall AUC | Better than ROC-AUC for imbalanced data |

### Model Handling Strategies

This model employs techniques to address class imbalance:

- ✅ **Stratified cross-validation**: Maintains class distribution in splits
- ✅ **Evaluation on minority class**: Prioritizes fraud detection metrics
- ✅ **Appropriate sampling**: Handles the 1:578 ratio during training
- ✅ **Cost-sensitive learning**: Penalizes misclassification of fraud more heavily

---

## Project Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                          🖥️ USER BROWSER                               │
│                                                                         │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 │ HTTP/HTTPS Requests
                                 │ JSON Payload
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│               📱 STREAMLIT FRONTEND (Port 8501)                         │
│          ┌──────────────────────────────────────────────────┐           │
│          │ • Transaction Input Interface                    │           │
│          │ • Feature Display (V1-V28)                       │           │
│          │ • Prediction Result Visualization                │           │
│          │ • Fraud Probability Display                      │           │
│          │ • Risk Level Indicators                          │           │
│          └──────────────────────────────────────────────────┘           │
│                                                                         │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
        POST /api/predict       GET /api/health
        POST /api/batch-predict GET /docs
                    │                         │
                    └────────────┬────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│            🚀 FASTAPI BACKEND (Port 8000)                              │
│        ┌───────────────────────────────────────────────────┐            │
│        │ Request Routing & Async Handling                 │            │
│        └────────────────────┬────────────────────────────┘             │
│                             │                                          │
│                             ▼                                          │
│        ┌───────────────────────────────────────────────────┐            │
│        │ 🔍 PYDANTIC VALIDATION LAYER                     │            │
│        │ • Input schema validation                        │            │
│        │ • Type checking                                  │            │
│        │ • Error handling                                 │            │
│        └────────────────────┬────────────────────────────┘             │
│                             │                                          │
│                             ▼                                          │
│        ┌───────────────────────────────────────────────────┐            │
│        │ 🔧 PREPROCESSING LAYER                           │            │
│        │ • Standardization (if applicable)                │            │
│        │ • Feature ordering                               │            │
│        │ • Missing value handling                         │            │
│        └────────────────────┬────────────────────────────┘             │
│                             │                                          │
│                             ▼                                          │
│        ┌───────────────────────────────────────────────────┐            │
│        │ 🧠 LOADED PICKLE MODEL                           │            │
│        │ • Scikit-Learn Classifier                        │            │
│        │ • Binary Classification                          │            │
│        │ • Probability Output (0-1)                       │            │
│        └────────────────────┬────────────────────────────┘             │
│                             │                                          │
│                             ▼                                          │
│        ┌───────────────────────────────────────────────────┐            │
│        │ 📊 RESPONSE FORMATTER                            │            │
│        │ • Prediction: 0 or 1                             │            │
│        │ • Probability: float(0-1)                        │            │
│        │ • Risk Level: Low/Medium/High                    │            │
│        │ • Confidence Score                               │            │
│        └────────────────────┬────────────────────────────┘             │
│                             │                                          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 │ JSON Response
                                 │ {prediction, probability, risk_level}
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│               📱 STREAMLIT FRONTEND - RESULT DISPLAY                    │
│          ┌──────────────────────────────────────────────────┐           │
│          │ ✅ LEGITIMATE / ⚠️ FRAUDULENT                   │           │
│          │ Probability: X%                                 │           │
│          │ Risk Level: [Color-coded indicator]             │           │
│          │ Confidence: X%                                  │           │
│          └──────────────────────────────────────────────────┘           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Through the System

```
1. User selects/enters transaction features
        ↓
2. Streamlit frontend sends HTTP POST request with JSON payload
        ↓
3. FastAPI receives request and routes to prediction endpoint
        ↓
4. Pydantic validates input schema
        ↓
5. Preprocessing layer prepares features for model
        ↓
6. Pickle-loaded Scikit-Learn model performs inference
        ↓
7. Raw prediction (0 or 1) converted to probability and risk level
        ↓
8. FastAPI formats response as JSON
        ↓
9. Streamlit receives response and displays results
        ↓
10. User sees prediction with confidence and risk indicators
```

---

## ✨ Features

### 🔍 Core Prediction Features

- **Single Transaction Prediction**: Real-time classification of individual transactions
- **Fraud Probability Scoring**: Continuous probability scores (0-1) indicating fraud likelihood
- **Risk Level Classification**: Categorical risk levels (Low/Medium/High) for quick assessment
- **Confidence Metrics**: Transparency into prediction certainty
- **Sample Transaction Library**: Pre-loaded dataset samples for easy testing

### 🎨 User Interface Features

- **Clean Streamlit Dashboard**: Intuitive transaction input interface
- **Feature Display**: Shows all 30 features (Time, Amount, V1-V28)
- **Real-time Results**: Instant prediction feedback with color-coded indicators
- **Transaction History**: Track predictions made during session
- **Responsive Design**: Works on desktop and mobile browsers

### 🚀 Technical & API Features

- **RESTful API Endpoints**: Standard HTTP methods (GET, POST)
- **Async Request Handling**: FastAPI async/await for high concurrency
- **Automatic API Documentation**: Swagger UI and ReDoc at `/docs` and `/redoc`
- **Robust Error Handling**: Comprehensive validation and informative error messages
- **Request/Response Validation**: Pydantic schemas for type safety
- **Production-Ready Model**: Scikit-Learn classifier serialized with Pickle
- **Health Check Endpoint**: Monitor API availability and status

---

## Installation & Setup

### Prerequisites

- **Python 3.10+** on your system
- **Git** for version control
- **pip** package manager
- **Virtual environment** (venv, conda, or pipenv)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Credit_Card_Fraud_Detection_Project.git
cd Credit_Card_Fraud_Detection_Project
```

### Step 2: Create Virtual Environment

**Windows (PowerShell/Command Prompt):**
```bash
python -m venv projectenv
projectenv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv projectenv
source projectenv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Key Dependencies:**
```
fastapi==0.138.0
streamlit>=1.28.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
pydantic>=2.0.0
uvicorn[standard]>=0.23.0
requests>=2.31.0
```

### Step 4: Verify Installation

```bash
# Verify all packages installed
pip list | grep -E "fastapi|streamlit|scikit-learn|pandas|numpy"

# Test imports
python -c "import fastapi, streamlit, sklearn, pandas; print('✅ All packages imported successfully!')"
```

### Step 5: Download the Full Dataset (Optional)

The full training dataset (284,807 rows) is available on Kaggle:

```bash
# Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Download creditcard.csv and place in: app/data/creditcard.csv
```

The project includes sample data (982 rows) for testing without downloading the full dataset.

---

## Project Structure

```
Credit_Card_Fraud_Detection_Project/
│
├── 📄 README.md                    # Project documentation (this file)
├── 📄 requirements.txt             # Python dependencies
├── 📄 .gitignore                   # Git ignore rules
├── 📄 LICENSE                      # MIT License
│
├── 📁 app/                         # Main application code
│   ├── 📄 main.py                  # FastAPI application & endpoints
│   ├── 📄 appy.py                  # Streamlit frontend interface
│   │
│   ├── 📁 data/                    # Data module
│   │   ├── 📄 __init__.py
│   │   ├── 📄 data.py              # Data loading & preprocessing
│   │   ├── 📊 creditcard.csv       # Full dataset (download from Kaggle)
│   │   └── 📊 sample_data.csv      # Sample 982 rows for testing
│   │
│   ├── 📁 model/                   # ML model module
│   │   ├── 📄 __init__.py
│   │   ├── 🧠 model.pkl            # Trained Scikit-Learn model
│   │   ├── 📄 train.py             # Model training script
│   │   └── 📊 model_metrics.json   # Performance metrics
│   │
│   └── 📁 schemas/                 # Pydantic validation schemas
│       ├── 📄 __init__.py
│       └── 📄 schemas.py           # Request/response data models
│
├── 📁 projectenv/                  # Python virtual environment
│   ├── 📄 pyvenv.cfg
│   ├── 📁 Scripts/ (or bin/)
│   ├── 📁 Lib/ (or lib/)
│   └── 📁 Include/
│
├── 📁 .streamlit/                  # Streamlit configuration
│   └── 📄 config.toml              # Streamlit settings
│
├── 📁 .github/
│   └── 📁 workflows/               # CI/CD workflows
│       └── 📄 deploy.yml           # GitHub Actions config
│
├── 📄 Dockerfile                   # Container image config
├── 📄 docker-compose.yml           # Multi-container orchestration
├── 📄 .dockerignore                # Docker ignore rules
└── 📄 .env.example                 # Environment variables template
```

---

## API Endpoints

### Base URL
- **Local**: `http://localhost:8000`
- **Documentation**: `http://localhost:8000/docs` (Swagger UI)

### Endpoints Reference

| Method | Endpoint | Purpose | Request Body | Response Status |
|--------|----------|---------|--------------|-----------------|
| **POST** | `/api/predict` | Single transaction prediction | Transaction JSON | 200, 422 |
| **GET** | `/api/transactions` | List sample transactions | None | 200 |
| **GET** | `/api/transactions/{id}` | Get specific sample transaction | None | 200, 404 |
| **POST** | `/api/batch-predict` | Batch prediction (CSV) | CSV file | 200, 400, 422 |
| **GET** | `/api/health` | API health status | None | 200 |
| **GET** | `/docs` | Swagger UI documentation | None | 200 |
| **GET** | `/redoc` | ReDoc documentation | None | 200 |

---

## Running the Application Locally

### Terminal 1: Start FastAPI Backend

```bash
cd app
python main.py
```

**Expected Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

**API Available At:**
- Main API: `http://localhost:8000`
- Swagger Docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Terminal 2: Start Streamlit Frontend

```bash
streamlit run app/appy.py
```

**Expected Output:**
```
Streamlit app running at http://localhost:8501
```

**Access Streamlit UI:** Open browser to `http://localhost:8501`

### Verify Both Services Are Running

```bash
# Check backend
curl http://localhost:8000/api/health

# Response:
{"status": "healthy", "timestamp": "2026-06-21T10:30:45Z"}
```

---

## Example Request & Response

### Example 1: Single Transaction Prediction

#### Request (POST /api/predict)

```json
{
  "time": 0,
  "amount": 149.62,
  "v1": -1.3598071336738,
  "v2": -0.0727236390324,
  "v3": 2.36060841498296,
  "v4": 1.34155470441765,
  "v5": -0.3480809033255,
  "v6": 0.8627493104857,
  "v7": 0.461193565360644,
  "v8": -0.0236456092272,
  "v9": 0.79934304449437,
  "v10": -0.551599533260813,
  "v11": -0.617800855762348,
  "v12": -0.991389847235408,
  "v13": -0.31169541528724,
  "v14": 1.46817697209427,
  "v15": -0.470400525259738,
  "v16": 0.207971241707901,
  "v17": 0.024798338405038,
  "v18": 0.204139938715054,
  "v19": 0.502233259031606,
  "v20": 0.219991212854701,
  "v21": 0.215153558098701,
  "v22": 0.665436389235245,
  "v23": 0.460818537088292,
  "v24": 0.061457629569833,
  "v25": 0.027961038882654,
  "v26": 0.029539241806241,
  "v27": 0.007837328778344,
  "v28": 0.271450160456249
}
```

#### Response (200 OK)

```json
{
  "transaction_id": "txn_12345",
  "prediction": 0,
  "prediction_label": "Legitimate",
  "fraud_probability": 0.08,
  "legitimate_probability": 0.92,
  "confidence": 0.92,
  "risk_level": "Low",
  "timestamp": "2026-06-21T10:35:22Z",
  "processing_time_ms": 2.34
}
```

### Example 2: Fraudulent Transaction (High Risk)

#### Request

```json
{
  "time": 53132,
  "amount": 0.76,
  "v1": -1.2768,
  "v2": 0.6522,
  ...
}
```

#### Response

```json
{
  "transaction_id": "txn_67890",
  "prediction": 1,
  "prediction_label": "Fraudulent",
  "fraud_probability": 0.92,
  "legitimate_probability": 0.08,
  "confidence": 0.92,
  "risk_level": "High",
  "timestamp": "2026-06-21T10:36:15Z",
  "processing_time_ms": 2.45
}
```

### Risk Level Classification

| Risk Level | Fraud Probability | Interpretation |
|-----------|------------------|-----------------|
| 🟢 **Low** | 0.00 - 0.33 | Likely Legitimate |
| 🟡 **Medium** | 0.34 - 0.66 | Uncertain, Requires Review |
| 🔴 **High** | 0.67 - 1.00 | Likely Fraudulent |

---

## Deployment Possibilities

### Option 1: AWS EC2

```bash
# 1. Launch EC2 instance (Ubuntu 22.04)
# 2. SSH into instance
# 3. Clone repository
# 4. Set up virtual environment
# 5. Install dependencies
# 6. Run FastAPI with Uvicorn:

uvicorn app.main:app --host 0.0.0.0 --port 8000

# 7. Run Streamlit (separate instance or process):
streamlit run app/appy.py --server.port 8501

# 8. Configure security groups for ports 8000 and 8501
# 9. (Optional) Use Nginx as reverse proxy
```

### Option 2: Docker & Docker Compose

#### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000 8501

CMD ["sh", "-c", "python app/main.py & streamlit run app/appy.py --server.port 8501"]
```

#### Docker Compose

```yaml
version: '3.8'

services:
  fastapi:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
    command: python app/main.py

  streamlit:
    build: .
    ports:
      - "8501:8501"
    environment:
      - PYTHONUNBUFFERED=1
    command: streamlit run app/appy.py --server.port 8501
```

**Deploy:**
```bash
docker-compose up --build
```

### Option 3: Render

#### Backend (FastAPI)

1. Create account at [render.com](https://render.com)
2. Connect GitHub repository
3. Create Web Service:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

#### Frontend (Streamlit)

1. Create another Web Service:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app/appy.py --server.port=$PORT --server.address=0.0.0.0`

### Option 4: Heroku (Legacy)

```bash
heroku login
heroku create your-app-name
git push heroku main
```

### Option 5: Google Cloud Run

```bash
gcloud run deploy fraud-detection \
  --source . \
  --platform managed \
  --region us-central1 \
  --memory 512Mi
```

---

## Future Improvements

### 🎯 Phase 1: Short-Term (Weeks 1-2)

- [ ] Add model performance metrics dashboard (Precision, Recall, F1, ROC-AUC)
- [ ] Implement transaction filtering and sorting in Streamlit
- [ ] Create CSV export functionality for prediction results
- [ ] Add dark mode to Streamlit UI
- [ ] Implement API rate limiting
- [ ] Add request logging and monitoring

### 🚀 Phase 2: Medium-Term (Weeks 3-8)

- [ ] **Model Explainability**: LIME/SHAP for feature contribution analysis
- [ ] **Multi-Model Support**: Ensemble methods (Random Forest, XGBoost, LightGBM)
- [ ] **Data Drift Detection**: Monitor model performance degradation
- [ ] **Database Integration**: PostgreSQL/MongoDB for transaction history
- [ ] **User Authentication**: API key management and role-based access
- [ ] **Advanced Monitoring**: ELK Stack, Prometheus, Grafana
- [ ] **CI/CD Pipeline**: GitHub Actions for automated testing/deployment
- [ ] **Unit Tests**: Pytest coverage for critical functions

### 🌟 Phase 3: Long-Term Vision (3+ Months)

- [ ] **Real-Time Streaming**: Kafka/Spark for live transaction processing
- [ ] **Model Retraining Pipeline**: Automated retraining with new data
- [ ] **A/B Testing Framework**: Compare multiple models in production
- [ ] **Mobile Application**: React Native/Flutter mobile app
- [ ] **Advanced Analytics**: Business intelligence dashboards
- [ ] **Microservices**: Kubernetes orchestration
- [ ] **GraphQL API**: Alternative to REST endpoints
- [ ] **Feature Store**: Centralized feature management (Feast)
- [ ] **Model Registry**: MLflow for experiment tracking
- [ ] **Explainable AI**: SHAP values, partial dependence plots

---

## Limitations

### 🔴 Critical Limitations

1. **Dataset Anonymization**
   - Features are PCA-transformed and not human-interpretable
   - Original banking attributes are permanently hidden
   - Model cannot provide business-explainable predictions

2. **Model Non-Explainability**
   - Predictions identify patterns but not underlying causes
   - Feature importance cannot be mapped to business logic
   - Not suitable for regulatory reporting without feature mappings

3. **Research Use Only**
   - Designed for educational and research purposes
   - Should NOT be used for real financial decision-making
   - Lacks access to original transaction attributes

4. **Class Imbalance**
   - Dataset has 1:578 fraud-to-legitimate ratio
   - Model may miss rare fraud patterns
   - Evaluation requires careful metric selection

5. **Model Generalization**
   - Trained on historical data with specific patterns
   - May not detect novel fraud techniques
   - Requires periodic retraining and validation

### 🟡 Technical Limitations

1. **Performance Constraints**
   - Single-threaded Streamlit frontend (no concurrent users)
   - Pickle format for model serialization (Python-only)
   - No horizontal scaling without additional infrastructure

2. **Data Handling**
   - Sample data limited to 982 rows
   - Full dataset requires ~500MB of storage
   - No support for real-time data streaming

3. **Monitoring**
   - Limited logging and observability
   - No alerting system for model drift
   - Manual intervention required for updates

---

## Disclaimer

### ⚠️ LEGAL AND USAGE DISCLAIMER

**This project uses an anonymized and PCA-transformed dataset published for research purposes.**

The predictions generated by this application:

1. ✅ **Identify statistical patterns** associated with fraudulent transactions
2. ✅ **Demonstrate machine learning engineering** and full-stack development
3. ✅ **Serve educational purposes** for learning ML model deployment
4. ❌ **CANNOT explain business reasons** for predictions (due to feature anonymization)
5. ❌ **SHOULD NOT be used** for real financial decision-making
6. ❌ **DO NOT provide explainability** required for regulatory compliance

### Not Suitable For:

- ❌ Production fraud detection systems
- ❌ Regulatory reporting (PCI-DSS, GDPR compliance)
- ❌ Real-world financial decisions
- ❌ Customer communication regarding fraud
- ❌ Audit trails and compliance documentation

### Recommended Usage:

✅ Portfolio demonstration of ML engineering skills  
✅ Learning full-stack ML development  
✅ Understanding model deployment architecture  
✅ Research and academic purposes  
✅ Proof-of-concept development  

### Liability Statement

**The authors and contributors of this project are NOT responsible for:**
- Misuse of the model for production fraud detection
- Financial losses from relying on model predictions
- Regulatory violations from using non-explainable models
- Data privacy breaches or unauthorized access

**Users must:**
- Understand the limitations of anonymized/PCA-transformed data
- Obtain proper legal and compliance guidance before deployment
- Validate predictions with original transaction features
- Maintain audit trails and documentation for any production use

---

## 📄 License

This project is licensed under the **MIT License**.

### You Are Free To:
- ✅ Use commercially or personally
- ✅ Modify and redistribute
- ✅ Include in proprietary projects

### You Must:
- ⚠️ Include the original license and copyright notice
- ⚠️ Provide a copy of the MIT License

See [LICENSE](LICENSE) file for full details.

---

## Acknowledgements

### Dataset Source
- **[ULB Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)** on Kaggle
- Dataset creators: Université Libre de Bruxelles (ULB)
- Publication: Andrea Dal Pozzolo, et al. (2015)

### Open Source Libraries
- **[Scikit-Learn](https://scikit-learn.org/)** - Machine learning framework
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern Python web framework
- **[Streamlit](https://streamlit.io/)** - Interactive web application framework
- **[Pandas](https://pandas.pydata.org/)** - Data manipulation library
- **[NumPy](https://numpy.org/)** - Numerical computing library
- **[Pydantic](https://docs.pydantic.dev/)** - Data validation library

### Community & Contributors
- Open source community for continuous improvements
- ML research community for best practices
- Stack Overflow and GitHub communities for troubleshooting

---

## 📞 Support & Contributing

### Report Issues
Open an [Issue](https://github.com/yourusername/Credit_Card_Fraud_Detection_Project/issues) for:
- Bug reports
- Feature requests
- Documentation improvements
- Questions about the project

### Submit Improvements
Create a [Pull Request](https://github.com/yourusername/Credit_Card_Fraud_Detection_Project/pulls) for:
- Bug fixes
- New features
- Code optimization
- Documentation enhancements

### Contact
- 📧 Email: your.email@example.com
- 💼 LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- 🐙 GitHub: [github.com/yourusername](https://github.com/yourusername)

---

<div align="center">

### 🌟 Appreciation

If this project helped you understand fraud detection modeling or ML deployment, please consider:
- ⭐ **Starring the repository**
- 🔗 **Sharing with others**
- 💬 **Providing feedback**
- 🤝 **Contributing improvements**

### ❤️ Made with dedication to ML engineering excellence

**Last Updated:** June 2026  
**Version:** 1.0.0  
**Status:** Production Ready (Research Purposes)

</div>

---

## Quick Reference

### Command Cheat Sheet

```bash
# Setup
python -m venv projectenv
projectenv\Scripts\Activate.ps1        # Windows
source projectenv/bin/activate         # macOS/Linux
pip install -r requirements.txt

# Run locally
python app/main.py                     # Terminal 1: FastAPI
streamlit run app/appy.py              # Terminal 2: Streamlit

# Docker
docker-compose up --build

# Health check
curl http://localhost:8000/api/health

# API Documentation
# Swagger UI: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
```

### API Response Examples

**Legitimate Transaction:**
```json
{
  "prediction": 0,
  "prediction_label": "Legitimate",
  "fraud_probability": 0.12,
  "risk_level": "Low"
}
```

**Fraudulent Transaction:**
```json
{
  "prediction": 1,
  "prediction_label": "Fraudulent",
  "fraud_probability": 0.87,
  "risk_level": "High"
}
```

---

*This README is part of the Credit Card Fraud Detection System portfolio project. For questions or contributions, please visit the [GitHub repository](https://github.com/yourusername/Credit_Card_Fraud_Detection_Project).*
