import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

# ── Tenure bins 
TENURE_BINS   = [0, 12, 24, 36, 48, 60, 72]
TENURE_LABELS = ['1-12', '13-24', '25-36', '37-48', '49-60', '61-72']

FEATURES = [
    'SeniorCitizen', 'MonthlyCharges', 'TotalCharges',
    'gender_Male',
    'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year',
    'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check',
    'tenure_bin_13-24', 'tenure_bin_25-36',
    'tenure_bin_37-48', 'tenure_bin_49-60', 'tenure_bin_61-72',
]


class DataProcessing:
    def __init__(self, input_path: str, output_path: str):
        self.input_path  = input_path
        self.output_path = output_path
        self.df          = None

        os.makedirs(self.output_path, exist_ok=True)
        logger.info("DataProcessing initialized.")

    # ── Step 1: load 
    def load_data(self):
        try:
            self.df = pd.read_csv(self.input_path)
            logger.info(f"Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} cols.")
        except Exception as e:
            logger.error(f"Load error: {e}")
            raise CustomException("Failed to load data", e)

    # ── Step 2: clean + feature-engineer 
    def preprocess(self):
        try:
           
            self.df['TotalCharges'] = pd.to_numeric(
                self.df['TotalCharges'], errors='coerce'
            )
            self.df.dropna(how='any', inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            logger.info(f"After cleaning: {len(self.df)} rows.")

            
            self.df['tenure_bin'] = pd.cut(
                self.df['tenure'],
                bins=TENURE_BINS,
                labels=TENURE_LABELS,
                include_lowest=True
            )
            logger.info("tenure_bin created.")
            logger.info(self.df['tenure_bin'].value_counts().sort_index().to_string())

            
            self.df['Churn'] = self.df['Churn'].map({'No': 0, 'Yes': 1})

           
            self.df.drop(columns=['customerID', 'tenure'], inplace=True)

            # --- one-hot encode everything 
            self.df = pd.get_dummies(
                self.df,
                columns=[
                    'gender', 'Partner', 'Dependents', 'PhoneService',
                    'MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies', 'Contract',
                    'PaperlessBilling', 'PaymentMethod', 'tenure_bin'
                ],
                drop_first=True   
            )

            # Convert bool columns produced by get_dummies to int
            bool_cols = self.df.select_dtypes(include='bool').columns
            self.df[bool_cols] = self.df[bool_cols].astype(int)

            logger.info(f"After encoding: {self.df.shape[1]} cols.")
            logger.info("Preprocessing complete.")

        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            raise CustomException("Failed to preprocess data", e)

    # ── Step 3: split → scale → save
    def split_scale_save(self):
        try:
            # Align: fill any missing feature cols with 0
            for col in FEATURES:
                if col not in self.df.columns:
                    self.df[col] = 0

            X = self.df[FEATURES]
            y = self.df['Churn']

            scaler   = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )

            joblib.dump(X_train,  os.path.join(self.output_path, 'X_train.pkl'))
            joblib.dump(X_test,   os.path.join(self.output_path, 'X_test.pkl'))
            joblib.dump(y_train,  os.path.join(self.output_path, 'y_train.pkl'))
            joblib.dump(y_test,   os.path.join(self.output_path, 'y_test.pkl'))
            joblib.dump(scaler,   os.path.join(self.output_path, 'scaler.pkl'))
            joblib.dump(FEATURES, os.path.join(self.output_path, 'features.pkl'))

            logger.info(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")
            logger.info("All artifacts saved.")

        except Exception as e:
            logger.error(f"Split/scale/save error: {e}")
            raise CustomException("Failed to split, scale, and save data", e)

    def run(self):
        self.load_data()
        self.preprocess()
        self.split_scale_save()


if __name__ == "__main__":
    processor = DataProcessing("artifacts/raw/data.csv", "artifacts/processed")
    processor.run()
