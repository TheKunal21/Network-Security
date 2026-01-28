import os 
import sys 
import json 

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")    
print("MONGO_DB_URL:", MONGO_DB_URL)

import certifi
ca = certifi.where()

import pandas as pd
import numpy as np
import pymongo
from Networksecurity.Exception.exception import NetworkSecurityException
from Networksecurity.Logging.logger import logging

class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def cv_to_json(self, file_path:str)->json:
        """
        file_path: str : path of the csv file 
        return : json data 
        """
        try:
            df=pd.read_csv(file_path)
            df.reset_index(drop=True, inplace=True)
            records = list(json.loads(df.T.to_json()).values())
            logging.info(f"csv file read successfully from path {file_path} ")
            return records
            return json.loads(df.to_json(orient='records'))
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def insert_data_mongodb(self,records,database,collection):
        """
        records : json data 
        database : database name 
        collection : collection name 
        """
        try:
            self.database = database
            self.collection = collection
            self.records = records
            
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return(len(self.records))
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
        
if __name__ == "__main__":
    FILE_PATH = 'Network_Data\PhisingData.csv'
    DATABASE = 'KUNAL_DB'
    COLLECTION = 'NetworkData'
    networkobj=NetworkDataExtract()
    records=networkobj.cv_to_json(file_path=FILE_PATH)
    print(records)
    number_of_records=networkobj.insert_data_mongodb(records=records,database=DATABASE,collection=COLLECTION)
    print(f"Total records inserted to Mongodb : {number_of_records}")