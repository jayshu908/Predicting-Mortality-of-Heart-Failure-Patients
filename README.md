# Predicting Mortality of Heart Failure Patients 🫀

This project, completed as part of my internship at **1Stop**, focuses on building a machine learning model to predict mortality events in patients suffering from heart failure. It aims to assist healthcare professionals in identifying high-risk patients and improving clinical decisions.

## 📌 Objective

To develop a predictive model using clinical data that can accurately identify whether a patient with heart failure is at risk of death during follow-up.

---

## 🗂 Dataset

- **Source**: [UCI Machine Learning Repository - Heart Failure Clinical Records Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records)
- **Samples**: 299 patients
- **Features**: 13 clinical features (age, sex, ejection fraction, serum creatinine, etc.)
- **Target**: `DEATH_EVENT` (0 = Alive, 1 = Deceased)

---

## 🛠️ Tech Stack

- **Language**: Python
- **Environment**: Google Colab
- **Libraries**:
  - `pandas`, `numpy` – Data manipulation
  - `matplotlib`, `seaborn` – Data visualization
  - `scikit-learn` – Model building
  - `xgboost` – Advanced modeling
  - `joblib` – Model saving

---

## 🔍 Workflow

1. **Exploratory Data Analysis**
   - Correlation matrix
   - Histograms, box plots
   - Outlier detection

2. **Data Preprocessing**
   - Missing value check
   - Feature scaling
   - Train-test split

3. **Model Training**
   - Logistic Regression
   - Random Forest
   - XGBoost
   - Hyperparameter tuning with GridSearchCV

4. **Evaluation Metrics**
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix
   - ROC-AUC Curve

5. **Model Persistence**
   - Saving the final model with `joblib` for future deployment

---

## 📈 Results

| Model             | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | XX%     | XX%       | XX%    | XX%      | XX%     |
| Random Forest       | XX%     | XX%       | XX%    | XX%      | XX%     |
| XGBoost             | **XX%** | **XX%**   | **XX%**| **XX%**  | **XX%** |

*Note: Replace XX% with your actual results*

---

## 📌 Key Takeaways

- Proper feature selection and preprocessing significantly affect model performance.
- XGBoost performed best in terms of generalization.
- The model can be integrated into healthcare systems to flag high-risk patients early.

---

## 💡 Future Improvements

- Try deep learning techniques with Keras or PyTorch.
- Integrate model into a web app using Flask or Streamlit.
- Work with real-time datasets from hospitals (if available).

---

## 📬 Contact

**Jayshree Jena**  
📧 jenajayshree2005@gmail.com  
🔗 https://www.linkedin.com/in/jayshree-jena
💻 https://github.com/jayshu908/

---

