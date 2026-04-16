# 🏦 Loan Defaulter Prediction
### 👤 Author: Sujal Singh - Data Scientist and ML / AI Engineer
# 📌 Project Overview
This project focuses on predicting whether a home loan applicant is likely to default using machine learning. By analyzing customer demographics, financial history, and credit behavior, the model assists banks in reducing credit risk and improving loan approval accuracy.

# 🎯 Business Problem
Loan defaults cause significant financial loss to institutions.Early detection of high‑risk applicants enables:
##### *Better risk assessment
##### *Effective credit policy planning
##### *Improved profit margins

# 📊 Key Features
##### *Data Size:Multiple datasets with demographics, credit risk indicators
##### *Classification Type:Binary classification (Default / Not Default)
##### *Primary Metric:Recall (detecting true defaulters)
##### *Deployment:Streamlit UI + FastAPI API
##### *Model Monitoring:Logging & threshold tuning included

# 🧹 Data Preprocessing
##### ✔ Outlier detection and capping
##### ✔ Missing values handled via median/mode + new category for >30% missing
##### ✔ Label & frequency encoding applied to categorical variables
##### ✔ Feature engineering based on correlation metrics
##### ✔ Standard scaling for numerical stability

# ⚙ Model & Evaluation
#### Multiple models were tested:
##### *Random Forest
##### *Gradient Boosting
##### *XGBoost
##### *EasyEnsembleClassifier (Best Performer)
# 🏆 Final Model Performance
##### ROC-AUC ≈ 0.64
##### Improved recall using threshold tuning
