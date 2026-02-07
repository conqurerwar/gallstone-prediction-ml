# 🩺 Gallstone Prediction System using Machine Learning

This project presents a machine learning–based predictive analytics system for detecting the presence of gallstones. It includes exploratory data analysis (EDA), multiple machine learning models, and an interactive Streamlit web interface for user input and prediction.

---

## 📌 Project Overview

Gallstones are a common health condition influenced by demographic, clinical, and biochemical factors. This project analyzes patient data and applies machine learning techniques to accurately predict gallstone presence.

The system allows users to input patient details and receive:
- Gallstone prediction result
- Best model accuracy
- Comparison of multiple ML models

---

## 🗂 Dataset Information

- Total Records: **319**
- Total Features: **38**
- Target Variable: **Gallstone Status**
  - `0` → No Gallstone
  - `1` → Gallstone Present
- Missing Values: **None**

---

## ⚙️ Technologies Used

- Python  
- Pandas & NumPy  
- Scikit-learn  
- Streamlit  
- Git & GitHub  

---

## 🤖 Machine Learning Models Implemented

- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- K-Nearest Neighbors (KNN)  

---

## 📊 Model Accuracy Comparison

| Model | Accuracy |
|------|----------|
| Logistic Regression | 78.12% |
| Decision Tree | 65.62% |
| Random Forest | **81.25%** |
| KNN | 68.75% |

🏆 **Best Performing Model:** Random Forest Classifier

---

## 🖥️ Streamlit Web Interface

The Streamlit application provides:
- Input fields for patient medical parameters
- Prediction of gallstone presence
- Display of trained model accuracy
- Accuracy comparison table of ML models

---

## ▶️ How to Run the Project

### Step 1: Install dependencies
```bash
pip install streamlit pandas scikit-learn

streamlit run app.py

http://localhost:8501

```
## Project Struture
├── app.py
├── gallstone (1).csv
├── README.md
├── .gitignore
├── LICENSE

