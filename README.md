# Health Condition Prediction Model | Machine Learning, Python, Streamlit

A machine learning model that predicts the likelihood of diabetes based on health data like glucose level, BMI, age, and blood pressure. The project involves data preprocessing, model training (Logistic Regression, Random Forest, etc.), and evaluation using accuracy, precision, and recall. Built with Streamlit for an interactive interface, it allows users to input health parameters and get real-time predictions, helping in early diabetes detection and preventive care.

## Features
- Uses Logistic regression
- Predicts health conditions from medical data
- Uses feature engineering to improve accuracy
- Interactive web app for user-friendly predictions
- Evaluates model performance with standard metrics
- I deployed it using Streamlit, which provides a simple UI for users to input their health details. The model runs in the backend and shows predictions in real-time.

## Tools Used
numpy
pandas
scikit-learn
streamlit
matplotlib
seaborn
plotly
joblib
st-aggrid

## Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

2. Run the app:
   streamlit run app2.py

### Files

├── app2.py                    # Streamlit app
├── feature_engineering.py     # Feature preprocessing
├── model_training_logreg.py   # Model training script
├── model_comparison.py        # Model evaluation/comparison
├── diabetes_model.pkl         # Saved model file
├── scaler.pkl                 # Saved scaler
├── encoders.pkl               # Saved encoders
├── diabetes_prediction_dataset.csv  # Dataset
├── requirements.txt           # Python dependencies
