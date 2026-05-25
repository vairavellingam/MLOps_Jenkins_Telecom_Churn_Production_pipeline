from src.data_processing import DataProcessing
from src.model_training import ModelTraining
from src.logger import get_logger

logger = get_logger("training_pipeline")

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("TELECOM CHURN - TRAINING PIPELINE STARTED")
    logger.info("=" * 60)

    logger.info("Step 1: Data Processing")
    processor = DataProcessing("artifacts/raw/data.csv", "artifacts/processed")
    processor.run()

    logger.info("Step 2: Model Training")
    trainer = ModelTraining("artifacts/processed/", "artifacts/models/")
    trainer.run()

    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 60)
