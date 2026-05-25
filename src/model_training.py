import os
import json
import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)


class ModelTraining:
    def __init__(self, processed_data_path: str, model_output_path: str):
        self.processed_path = processed_data_path
        self.model_path = model_output_path
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.ada_model = None
        self.xgb_model = None

        os.makedirs(self.model_path, exist_ok=True)
        logger.info("ModelTraining initialized.")

    def load_data(self):
        try:
            self.X_train = joblib.load(os.path.join(self.processed_path, 'X_train.pkl'))
            self.X_test  = joblib.load(os.path.join(self.processed_path, 'X_test.pkl'))
            self.y_train = joblib.load(os.path.join(self.processed_path, 'y_train.pkl'))
            self.y_test  = joblib.load(os.path.join(self.processed_path, 'y_test.pkl'))
            logger.info("Processed data loaded successfully.")
        except Exception as e:
            logger.error(f"Load error: {e}")
            raise CustomException("Failed to load processed data", e)
        
    def apply_smote(self):
        try:
            logger.info("Applying SMOTE to training data...")
            smote = SMOTE(random_state=42)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            logger.info(f"After SMOTE: {self.X_train.shape[0]} samples.")
        except Exception as e:
            logger.error(f"SMOTE error: {e}")
            raise CustomException("Failed to apply SMOTE", e)

    def train_adaboost(self):
        try:
            logger.info("Training AdaBoost model...")
            base_estimator = DecisionTreeClassifier(max_depth=1)
            self.ada_model = AdaBoostClassifier(
                estimator=base_estimator,
                n_estimators=200,
                learning_rate=0.5,
                random_state=42
            )
            self.ada_model.fit(self.X_train, self.y_train)
            joblib.dump(self.ada_model, os.path.join(self.model_path, 'adaboost_model.pkl'))
            logger.info("AdaBoost trained and saved.")
        except Exception as e:
            logger.error(f"AdaBoost training error: {e}")
            raise CustomException("Failed to train AdaBoost", e)

    def train_xgboost(self):
        try:
            logger.info("Training XGBoost model...")
            self.xgb_model = XGBClassifier(
                 n_estimators=177,
                 max_depth=4,
                 learning_rate=0.0278,
                 subsample=0.655,
                 colsample_bytree=0.655,
                 min_child_weight=10,
                 gamma=0.476,
                 reg_alpha=0.093,
                 reg_lambda=0.983,
                 eval_metric="logloss",
                 random_state=42
                 )
            self.xgb_model.fit(self.X_train, self.y_train)
            joblib.dump(self.xgb_model, os.path.join(self.model_path, 'xgboost_model.pkl'))
            logger.info("XGBoost trained and saved.")
        except Exception as e:
            logger.error(f"XGBoost training error: {e}")
            raise CustomException("Failed to train XGBoost", e)

    def evaluate_model(self, model, name: str) -> dict:
        try:
            y_pred  = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1]

            metrics = {
                "model":     name,
                "accuracy":  round(accuracy_score(self.y_test, y_pred),  4),
                "precision": round(precision_score(self.y_test, y_pred), 4),
                "recall":    round(recall_score(self.y_test, y_pred),    4),
                "f1_score":  round(f1_score(self.y_test, y_pred),        4),
                "roc_auc":   round(roc_auc_score(self.y_test, y_proba),  4),
            }

            logger.info(f"[{name}] Accuracy={metrics['accuracy']} | "
                        f"Precision={metrics['precision']} | "
                        f"Recall={metrics['recall']} | "
                        f"F1={metrics['f1_score']} | "
                        f"ROC-AUC={metrics['roc_auc']}")

            logger.info(f"\n{classification_report(self.y_test, y_pred)}")
            return metrics
        except Exception as e:
            logger.error(f"Evaluation error for {name}: {e}")
            raise CustomException(f"Failed to evaluate {name}", e)

    def save_metrics(self, ada_metrics: dict, xgb_metrics: dict):
        metrics_path = os.path.join(self.model_path, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump({"adaboost": ada_metrics, "xgboost": xgb_metrics}, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")

    def run(self):
        self.load_data()
        self.apply_smote()
        self.train_adaboost()
        self.train_xgboost()
        ada_metrics = self.evaluate_model(self.ada_model, "AdaBoost")
        xgb_metrics = self.evaluate_model(self.xgb_model, "XGBoost")
        self.save_metrics(ada_metrics, xgb_metrics)
        logger.info("Model training pipeline complete.")


if __name__ == "__main__":
    trainer = ModelTraining("artifacts/processed/", "artifacts/models/")
    trainer.run()
