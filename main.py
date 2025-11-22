import sys
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException

from networksecurity.pipeline.training_pipeline import TrainingPipeline


def main():
    try:
        logging.info("========== STARTING NETWORK SECURITY TRAINING PIPELINE ==========")

        pipeline = TrainingPipeline()

        # Run the entire workflow
        model_trainer_artifact = pipeline.run_pipeline()
        logging.info(f"ModelTrainerArtifact: {model_trainer_artifact}")

        # Upload artifacts to S3
        pipeline.sync_artifact_dir_to_s3()
        pipeline.sync_saved_model_dir_to_s3()

        logging.info("========== TRAINING COMPLETED & SYNCED TO S3 ==========")

    except Exception as e:
        raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    main()
