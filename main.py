from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.entity.config_entity import DataIngestionConfig,DataValidationConfig
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import TrainingPipelineConfig,DataValidationConfig,DataTransformationConfig
from networksecurity.components.data_validation import DataValidation
import sys



if __name__ == '__main__':
    try:
        trainingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        dataingestion = DataIngestion(dataingestionconfig)
        logging.info("Data Initiation")
        dataingestionartifact = dataingestion.initiate_data_ingestion()
        logging.info("Data Initiation completed")
        datavalidationconfig = DataValidationConfig(trainingpipelineconfig)
        data_validation = DataValidation(dataingestionartifact,datavalidationconfig)
        
        data_validation_artifact=data_validation.initiate_data_validation()
        print(dataingestionartifact)
        
        data_transformation_config =DataTransformationConfig(trainingpipelineconfig)
        data_transformation= DataTransformation(data_validation_artifact, data_transformation_config)
        data_transformation_artifact= data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)
        
        
    except Exception as e:
        raise NetworkSecurityException(e,sys)