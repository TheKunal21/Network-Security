from sklearn.metrics import r2_score
import yaml 
from Networksecurity.Exception.exception import NetworkSecurityException
from Networksecurity.Logging.logger import logging
import sys
import os
#import dill
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV

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
    

def save_numpy_array_data(file_path:str,array:np.array):
    
    """
    file_path : str : path to file 
    array : np.array : numpy array data to be saved 
    return : None 
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path,'wb') as file_obj:
            np.save(file_obj,array)
        logging.info(f"Numpy array saved successfully at {file_path}")
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
    
def save_object(file_path:str,obj:object)->None:
    """
    file_path : str : path to file 
    obj : object : object to be saved 
    return : None 
    """
    try:
        logging.info(f"Entered the save_object method of utils class")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
        logging.info(f"Object saved successfully at {file_path} & Exited the save_object of MainUtils class")
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
    
def load_object(file_path:str)->object:
    """
    file_path : str : path to file 
    return : object : loaded object 
    """
    try:
       if not os.path.exists(file_path):
           raise Exception(f"The file: {file_path} is not exists")
       with open(file_path,'rb') as file_obj:
           print(file_obj)
           return pickle.load(file_obj)
         
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    

def load_numpy_array_data(file_path:str)->np.array:
    """
    file_path : str : path to file 
    return : np.array : loaded numpy array 
    """
    try:
        logging.info(f"Loading numpy array data from file: {file_path}")
        with open(file_path,'rb') as file_obj:
            return np.load(file_obj) 
        logging.info(f"Numpy array loaded successfully from {file_path}")
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
    
def evaluate_models(X_train,y_train,X_test,y_test,models:dict,params:dict)->dict:
    """
    X_train : np.array : training feature array 
    y_train : np.array : training label array 
    X_test : np.array : testing feature array 
    y_test : np.array : testing label array 
    models : dict : dictionary of models to be evaluated 
    params : dict : dictionary of parameters for each model 
    return : dict : dictionary of model name and its r2 score 
    """
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]
            #print(f"Model Name: {list(models.keys())[i]}")
            
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            
            # model.fit(X_train,y_train) # training the model
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report
    
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
    
