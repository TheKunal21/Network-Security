from Networksecurity.Components.data_ingestion import DataIngestion
from Networksecurity.Exception.exception import NetworkSecurityException
from Networksecurity.Logging.logger import logging
import sys
from Networksecurity.entity.config_entity import DataIngestionConfig
from Networksecurity.entity.config_entity import TrainingPipelineConfig


if __name__ == "__main__":
    try:
        trainingpipelineconfig=TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        dataingestion = DataIngestion(dataingestionconfig)
        logging.info('intitate the data ingestion')
        dataingestionartifact =dataingestion.initiate_data_ingestion()
        print(dataingestionartifact)
    except Exception as e :
        raise NetworkSecurityException(e,sys)