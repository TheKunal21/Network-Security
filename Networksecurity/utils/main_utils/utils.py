import yaml 
from Networksecurity.Exception.exception import NetworkSecurityException
from Networksecurity.Logging.logger import logging
import sys
import os
#import dill
import numpy as np
import pickle

def read_yaml_file(file_path:str)->dict:
    """
    file_path : str : path to yaml file 
    return : dict : dictionary of yaml file content 
    """
    try:
        with open(file_path,'rb') as yaml_file:
            content = yaml.safe_load(yaml_file)
        logging.info(f"yaml file: {file_path} loaded successfully")
        return content
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def write_yaml_file(file_path:str,content:dict,replace:bool = False)->None:
    """
    file_path : str : path to yaml file 
    content : dict : content to be written in yaml file
    return : None
    """
    try:
       if replace:
           if os.path.exists(file_path):
               os.remove(file_path)
               
       os.makedirs(os.path.dirname(file_path), exist_ok=True)
       with open(file_path, 'w') as yaml_file:
           yaml.dump(content, yaml_file)
       logging.info(f"yaml file: {file_path} written successfully")
           
            
    except Exception as e:
        raise NetworkSecurityException(e, sys)