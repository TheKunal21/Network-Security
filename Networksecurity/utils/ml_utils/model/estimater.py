from Networksecurity.Constant.training_pipeline import SAVED_MODEL_DIR,  MODEL_FILE_NAME

import os
import sys

from Networksecurity.Exception.exception import NetworkSecurityException
from Networksecurity.Logging.logger import logging


class NetworkModel:
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise NetworkSecurityException(e, sys)


def predict(self,x):
        try:
            logging.info("Prediction started")
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            logging.info("Prediction completed")
            return y_hat
        except Exception as e:
            raise NetworkSecurityException(e, sys)