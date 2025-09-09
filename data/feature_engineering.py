# feature_engineering.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# ============================== Load and Prepare Data ==============================
def load_and_prepare_data(path):
    df = pd.read_csv(path)

    # ================== Encode categorical variables ==================
    le_gender = LabelEncoder()
    df['gender'] = le_gender.fit_transform(df['gender'])

    le_smoking = LabelEncoder()
    df['smoking_history'] = le_smoking.fit_transform(df['smoking_history'])

    encoders = {
        'gender': le_gender,
        'smoking_history': le_smoking
    }

    # ================== Scale features ==================
    X = df.drop('diabetes', axis=1)
    y = df['diabetes']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ================== Train-test split ==================
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler, encoders

# ============================== Save Encoders/Scaler for Inference ==============================
if __name__ == "__main__":
    data_path = r"D:\python eda\diabetes_prediction_dataset.csv"
    X_train, X_test, y_train, y_test, scaler, encoders = load_and_prepare_data(data_path)

    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(encoders, 'encoders.pkl')

    print("Scaler and encoders saved successfully.")
