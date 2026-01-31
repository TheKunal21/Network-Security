import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from Networksecurity.Constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS, TARGET_COLUMN
from Networksecurity.Exception.exception import NetworkSecurityException
from Networksecurity.Logging.logger import logging
from Networksecurity.entity.artifact_entity import DataTransformationArtifact,DataValidationArtifact
from Networksecurity.entity.config_entity import DataTransformationConfig
from Networksecurity.utils.main_utils.utils import save_numpy_array_data,save_object


class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                    data_transformation_config:DataTransformationConfig):  
        
        try:
            self.data_validation_artifact:DataValidationArtifact = data_validation_artifact
            self.data_transformation_config:DataTransformationConfig = data_transformation_config
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
        
    @staticmethod
    def read_data(file_path:str)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    @staticmethod
    def get_data_transformer_object()->Pipeline:
        
        """
        it initiates a KNN imputer object with the parameters specified in the training_pipeline.py
        file and returns a Pipeline object with the KNNImputer object as the 
        first step.
        
        Args:
            cls: DataTransformation 
            
        returns:
            Pipeline: sklearn Pipeline object with KNNImputer as the first step.
        
        Raises:
            NetworkSecurityException: _description_
        """
        logging.info("Entered Data Transformation initiated method or get_data_transformer_object")
        try:
            imputer: KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(f"initialise the KNNImputer with parameters: {DATA_TRANSFORMATION_IMPUTER_PARAMS}")
            preprocessor: Pipeline = Pipeline(steps=[
                ('imputer', imputer)
            ])
            return preprocessor
            
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info("Data Transformation started")
            
            #load training and testing data
            train_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_train_file_path
            )
            test_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_test_file_path  )
            logging.info("Train and Test data loaded successfully")
            
            #separating input features and target feature
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df=target_feature_train_df.replace(-1,0)
            
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df=target_feature_test_df.replace(-1,0)
            
            
            
            preprocessing_pipeline = self.get_data_transformer_object()
            
            #fitting and transforming training data
         
            
            preprocessor_object = preprocessing_pipeline.fit(input_feature_train_df)
            transformed_input_feature_train_arr = preprocessor_object.transform(input_feature_train_df)
            transformed_input_feature_test_arr = preprocessing_pipeline.transform(input_feature_test_df)
            
            #combining input features and target feature arrays
            train_arr = np.c_[transformed_input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_feature_test_arr, np.array(target_feature_test_df)]
            
            #saving transformed data
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr
            )
            
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )
            
            save_object(
                self.data_transformation_config.transformed_object_file_path,preprocessor_object
                
            )
            
            #preparing artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path = self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path = self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path = self.data_transformation_config.transformed_object_file_path
            )
            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)