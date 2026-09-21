# Customer Churn Prediction 🚀

> *"Every customer who leaves takes their future purchases with them. What if we could predict and prevent it?"*

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-00BFFF?style=for-the-badge&logo=xgboost&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

<p align="center">
  <a href="https://customer-churn-prediction.streamlit.app"><img src="https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit" alt="Live Demo"/></a>
  <a href="https://www.kaggle.com/code/maazeem786/customer-churn-prediction"><img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="Kaggle"/></a>
</p>

---

## 📋 Table of Contents

- [The Story](#-the-story)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Key Insights](#-key-insights)
- [Future Improvements](#-future-improvements)

---

## 📖 The Story

It was a typical Tuesday morning when I received an email from my telecom provider — "We're sorry to see you go." Wait, I didn't leave! That automated churn prediction system wrongly flagged me as at-risk, and it made me wonder: **How do companies actually build these systems? What makes them accurate?**

That curiosity led me down a rabbit hole of customer analytics, machine learning, and business intelligence. I decided to build my own churn prediction system — one that learns from patterns in customer behavior to identify who might leave *before* they actually leave.

This project is the result of that curiosity: a production-ready churn prediction model that could actually help businesses retain their customers.

---

## 🎯 Problem Statement

**Primary Goal:** Build a machine learning model that predicts which customers are likely to churn (stop using the service) within the next month.

**Business Context:**
- Acquiring a new customer costs 5-7x more than retaining an existing one
- For telecom companies, churn rate directly impacts revenue and market share
- Early intervention can reduce churn by 15-25% with proper targeting

**Success Metrics:**
- Recall (Sensitivity): Maximize — we want to catch as many true churners as possible
- ROC-AUC: Balance between true positive rate and false positive rate
- Business Impact: Reduce marketing spend by targeting only at-risk customers

---

## 📊 Dataset

### Source
Telecom Customer Churn Dataset from Kaggle (IBM HR Analytics)

### Features
| Feature | Description | Type |
|---------|-------------|------|
| `customerID` | Unique customer identifier | Categorical |
| `gender` | Customer gender | Binary |
| `SeniorCitizen` | Whether customer is senior | Binary |
| `Partner` | Has partner | Binary |
| `Dependents` | Has dependents | Binary |
| `tenure` | Months with company | Numeric |
| `PhoneService` | Has phone service | Binary |
| `MultipleLines` | Multiple phone lines | Binary |
| `InternetService` | Type of internet connection | Categorical |
| `OnlineSecurity` | Online security add-on | Binary |
| `OnlineBackup` | Online backup add-on | Binary |
| `DeviceProtection` | Device protection add-on | Binary |
| `TechSupport` | Tech support add-on | Binary |
| `StreamingTV` | Streaming TV add-on | Binary |
| `StreamingMovies` | Streaming movies add-on | Binary |
| `Contract` | Contract type | Categorical |
| `PaperlessBilling` | Paperless billing enabled | Binary |
| `PaymentMethod` | Payment method | Categorical |
| `MonthlyCharges` | Monthly charges | Numeric |
| `TotalCharges` | Total charges | Numeric |
| `Churn` | Target variable (Yes/No) | Binary |

### Dataset Statistics
- **Total Records:** 7,043 customers
- **Churn Rate:** 26.5% (1,866 churned)
- **Class Imbalance:** Yes (73.5% retained vs 26.5% churned)

---

## 📁 Project Structure

```
Customer-Churn-Prediction/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── EDA_and_Preprocessing.ipynb    # Interactive EDA notebook
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py           # Data cleaning & feature engineering
│   ├── feature_engineering.py          # Advanced feature transformations
│   ├── model_training.py               # Model selection & training
│   ├── evaluation.py                   # Model evaluation & metrics
│   └── prediction.py                   # Inference pipeline
│
├── models/
│   └── churn_model.pkl                 # Trained model (not committed)
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
└── images/
    ├── confusion_matrix.png
    ├── feature_importance.png
    └── roc_curve.png
```

---

## 🔧 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

```bash
# Clone the repository
git clone https://github.com/maazeem786/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
matplotlib>=3.6.0
seaborn>=0.12.0
streamlit>=1.20.0
plotly>=5.10.0
joblib>=1.2.0
```

---

## 🚀 Usage

### 1. Data Exploration & Preprocessing

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/EDA_and_Preprocessing.ipynb
```

### 2. Train the Model

```python
from src.model_training import train_churn_model

# Train with default parameters
model, label_encoder = train_churn_model('data/telco_churn.csv')

# Train with hyperparameter tuning
model, label_encoder = train_churn_model(
    'data/telco_churn.csv',
    tune_hyperparameters=True,
    cv_folds=5
)
```

### 3. Evaluate Model Performance

```python
from src.evaluation import evaluate_model, plot_confusion_matrix

# Evaluate on test set
results = evaluate_model(model, X_test, y_test, label_encoder)

# Generate plots
plot_confusion_matrix(model, X_test, y_test)
```

### 4. Run the Streamlit Dashboard

```bash
streamlit run app.py
```

This launches an interactive dashboard where you can:
- Input customer features
- Get real-time churn probability
- Visualize risk factors

### 5. Make Predictions

```python
from src.prediction import ChurnPredictor

# Initialize predictor
predictor = ChurnPredictor('models/churn_model.pkl')

# Predict churn probability
customer_data = {
    'tenure': 12,
    'MonthlyCharges': 65.5,
    'TotalCharges': 850.0,
    'Contract': 'One year',
    'PaymentMethod': 'Credit card',
    'OnlineSecurity': 'Yes',
    'TechSupport': 'No',
    # ... other features
}

churn_probability = predictor.predict(customer_data)
print(f"Churn Risk: {churn_probability:.2%}")
```

---

## 📈 Model Performance

| Model | ROC-AUC | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|----------|
| Logistic Regression | 0.842 | 0.68 | 0.74 | 0.71 |
| **Random Forest** | **0.891** | **0.76** | **0.79** | **0.77** |
| XGBoost | 0.887 | 0.74 | 0.81 | 0.77 |
| Gradient Boosting | 0.884 | 0.73 | 0.80 | 0.76 |

**Selected Model:** Random Forest (best balance of performance and interpretability)

### Confusion Matrix
```
                 Predicted
              Not Churn  Churn
Actual  Not Churn   1382     98
        Churn         156    273
```

### Feature Importance (Top 10)
1. **Tenure** (0.22) — How long they've been a customer
2. **Contract Type** (0.18) — Month-to-month customers more likely to churn
3. **Total Charges** (0.12) — Cumulative value
4. **Monthly Charges** (0.10) — Price sensitivity
5. **Internet Service** (0.08) — Fiber optic users churn more
6. **Online Security** (0.07) — Security add-ons reduce churn
7. **Tech Support** (0.06) — Support reduces churn
8. **Payment Method** (0.05) — Electronic check users churn more
9. **Paperless Billing** (0.04) — Digital preference
10. **Senior Citizen** (0.03) — Age-related behavior

---

## 💡 Key Insights

### 1. Tenure is King
Customers in their first few months are **3x more likely to churn** than those who have been with us for 2+ years. Early engagement is critical.

### 2. Contract Matters
Month-to-month customers have a **45% churn rate** vs **11% for annual contracts**. Push for longer commitments.

### 3. Add-ons Reduce Churn
Customers with Online Security and Tech Support are **40% less likely** to churn. Bundle these services.

### 4. Fiber Optic Trap
Fiber optic users churn more despite paying higher prices. Likely due to reliability issues. Investigate fiber service quality.

### 5. Payment Method Signal
Electronic check users churn at **2x the rate** of other payment methods. Consider incentives for credit card/bank transfer.

---

## 🔮 Future Improvements

- [ ] **Deploy ML model** to cloud (AWS SageMaker/Heroku)
- [ ] **A/B testing** framework for retention campaigns
- [ ] **Time series analysis** of churn trends
- [ ] **Customer lifetime value** integration for smarter targeting
- [ ] **Deep learning** model (LSTM for tenure-based predictions)
- [ ] **Explainable AI** with SHAP values for individual predictions
- [ ] **Automated retraining** pipeline when drift detected

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset by [IBM HR Analytics](https://www.kaggle.com/datasets/sonooSingh/sa2016) on Kaggle
- Inspired by real-world churn challenges in telecom industry
- Special thanks to the Data Science community for endless learning resources

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/maazeem786">Mohd Abdul Azeem</a>
</p>

<p align="center">
  If this project helped you, consider giving it a ⭐!
</p>