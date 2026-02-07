import os
import sys
from Networksecurity.Exception.exception import NetworkSecurityException
from Networksecurity.Logging.logger import logging
from Networksecurity.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifact
from Networksecurity.entity.config_entity import ModelTrainerConfig
from Networksecurity.utils.main_utils.utils import load_object  ,save_object , evaluate_models
from Networksecurity.utils.main_utils.utils import load_numpy_array_data,save_numpy_array_data
from Networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from Networksecurity.utils.ml_utils.model.estimater import NetworkModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
import mlflow

import dagshub
dagshub.init(repo_owner='TheKunal21', repo_name='Network-Security', mlflow=True)



class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,
                 data_transformation_artifact:DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            logging.error(f"Error in Model Trainer class init: {e}")
            raise NetworkSecurityException(e, sys)
    
    def track_mlflow(self,best_model,classificationmetric):
        with mlflow.start_run():
            f1_score = classificationmetric.f1_score
            precision_score = classificationmetric.precision_score
            recall_score=classificationmetric.recall_score
            
            mlflow.log_metric("f1_score",f1_score)
            mlflow.log_metric("precision",precision_score)
            mlflow.log_metric("recall",recall_score)
            mlflow.sklearn.log_model(best_model,"model")
            
    
    
    def train_model(self, X_train, y_train, X_test, y_test) -> NetworkModel:
        models = {
            "LogisticRegression": LogisticRegression(verbose=1),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "RandomForestClassifier": RandomForestClassifier(verbose=1),
            "AdaBoostClassifier": AdaBoostClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(verbose=1)
        }
        params = {
            "LogisticRegression": {},
            "RandomForestClassifier": {
                "n_estimators": [8, 16, 32, 50, 64, 100, 128, 256],
                "max_depth": [None, 10, 20],
                "criterion": ["gini", "entropy", "log_loss"],
                "max_features": ["sqrt", "log2"],
            },
            "DecisionTreeClassifier": {
                "criterion": ["gini", "entropy", "log_loss"],
                "splitter": ["best", "random"],
                "max_features": ["sqrt", "log2"],
            },
            "KNeighborsClassifier": {
                "n_neighbors": [3, 5, 7, 9, 11],
                "weights": ["uniform", "distance"],
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            },
            "AdaBoostClassifier": {
                "n_estimators": [8, 16, 32, 50, 64, 100, 128, 256],
                "learning_rate": [0.01, 0.1, 0.5, 1.0],
            },
            "GradientBoostingClassifier": {
                "criterion": ['friedman_mse', 'squared_error'],
                "learning_rate": [0.01, 0.1, 0.5, 1.0],
                "n_estimators": [8, 16, 32, 50, 64, 100, 128, 256],
                "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                "loss": ['log_loss', 'exponential'],
            }
        }
        model_report: dict = evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            models=models,
            params=params
        )
        
        ## TO get best model score from dict
        
        best_model_score = max(sorted(model_report.values()))
        
        ## To get best model name from dict 
        
        best_model_name = list(model_report.keys())[
            
            list(model_report.values()).index(best_model_score)
        ]
        
        best_model = models[best_model_name]
        y_train_pred = best_model.predict(X_train)
        
        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
        
        
        ## Track the experiments with  mlflow
        self.track_mlflow(best_model,classification_train_metric)
        
        
        y_test_pred = best_model.predict(X_test)
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)
        Network_Model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj=Network_Model)
        
        save_object("final_model/model.pkl", best_model)

        return ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric
        )
        
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Loading transformed training and testing arrays")
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # loading training array and testing array
            train_array = load_numpy_array_data(train_file_path)
            test_array = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            logging.info("Model training started")

            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            logging.info("Training the model")
            return model_trainer_artifact
        except Exception as e:
            logging.error(f"Error in initiate_model_trainer: {e}")
            raise NetworkSecurityException(e, sys)