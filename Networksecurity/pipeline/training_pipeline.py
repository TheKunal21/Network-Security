import os
import sys

from Networksecurity.Components import data_transformation
from Networksecurity.Logging.logger import logging
from Networksecurity.Exception.exception import NetworkSecurityException

from Networksecurity.Components.data_ingestion import DataIngestion
from Networksecurity.Components.data_validation import DataValidation
from Networksecurity.Components.data_transformation import DataTransformation
from Networksecurity.Components.model_trainer import ModelTrainer

from Networksecurity.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    TrainingPipelineConfig
)

from Networksecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)


class TrainingPipeline:
    def __init__(self):
       
            self.training_pipeline_config = TrainingPipelineConfig()
            
    def start_data_ingestion(self) -> DataIngestionArtifact:
                try:
                    data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
                    logging.info('Initiate the data ingestion')
                    data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
                    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
                    logging.info(f"Data Ingestion Completed : {data_ingestion_artifact}")
                    return data_ingestion_artifact
                except Exception as e:
                    raise NetworkSecurityException(e, sys)
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
                try:
                    data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
                    logging.info("Initiate the data validation")
                    data_validation = DataValidation(
                        data_validation_config=data_validation_config,
                        data_ingestion_artifact=data_ingestion_artifact
                    )
                    data_validation_artifact = data_validation.initiate_data_validation()
                    logging.info(f"Data validation completed : {data_validation_artifact}")
                    return data_validation_artifact
                except Exception as e:
                    raise NetworkSecurityException(e, sys)
    
    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
                try:
                    logging.info("Data transformation started")
                    data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
                    data_transformation = DataTransformation(
                        data_validation_artifact=data_validation_artifact,
                        data_transformation_config=data_transformation_config
                    )
                    data_transformation_artifact = data_transformation.initiate_data_transformation()
                    logging.info(f"Data Transformation completed : {data_transformation_artifact}")

                    return data_transformation_artifact
                except Exception as e:
                    raise NetworkSecurityException(e, sys)


    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
                try:
                    logging.info("Model Training started")
                    model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
                    model_trainer = ModelTrainer(
                        model_trainer_config=model_trainer_config,
                        data_transformation_artifact=data_transformation_artifact)
                    model_trainer_artifact = model_trainer.initiate_model_trainer()
                    logging.info(f"Model Training completed : {model_trainer_artifact}")
                    return model_trainer_artifact
                except Exception as e:
                    raise NetworkSecurityException(e, sys)
                        
                        
    def run_pipeline(self):
                try:
                    data_ingestion_artifact = self.start_data_ingestion()
                    data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
                    data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
                    model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
                    logging.info(f"Training pipeline completed successfully with Model Trainer Artifact : {model_trainer_artifact}")
                except Exception as e:
                    logging.error(f"Error in training pipeline: {e}")
                    raise NetworkSecurityException(e, sys)
                










