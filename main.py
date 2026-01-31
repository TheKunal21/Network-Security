from Networksecurity.Components.data_ingestion import DataIngestion
from Networksecurity.Exception.exception import NetworkSecurityException
from Networksecurity.Logging.logger import logging
import sys
from Networksecurity.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig
from Networksecurity.entity.config_entity import training_pipeline
from Networksecurity.entity.config_entity import TrainingPipelineConfig 
from Networksecurity.Components.data_validation import DataValidation
from Networksecurity.Components.data_transformation import DataTransformation

if __name__ == "__main__":
    try:
        # Create ONE shared config with ONE timestamp for the whole run
        trainingpipelineconfig = TrainingPipelineConfig()
        
        # 1. Data Ingestion
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        dataingestion = DataIngestion(dataingestionconfig)
        logging.info('Initiate the data ingestion')
        dataingestionartifact = dataingestion.initiate_data_ingestion()
        logging.info("Data Initiation Completed")
        
        # 2. Data Validation
        data_validation_config = DataValidationConfig(trainingpipelineconfig)
        
        # FIX: Ensure order matches __init__(self, config, artifact)
        data_validation = DataValidation(
            data_validation_config=data_validation_config,
            data_ingestion_artifact=dataingestionartifact
        )
        
        logging.info("Initiate the data validation")
        print(dataingestionartifact)
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data validation completed")
        print(data_validation_artifact)
        logging.info("Data transformation started")
        data_transformation_config = DataTransformationConfig(trainingpipelineconfig)
        data_transformation=DataTransformation(data_validation_artifact,data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)
        logging.info("Data Transformation completed")
    except Exception as e:
        logging.error(e)
        raise NetworkSecurityException(e, sys)