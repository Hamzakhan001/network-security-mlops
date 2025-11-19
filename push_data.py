import os
import sys
import json
from dotenv import load_dotenv
load_dotenv()
import certifi
ca = certifi.where()
import pandas as pd
import numpy as np
import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging



MONGO_DB_URL = os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)

if not MONGO_DB_URL:
    raise NetworkSecurityException(ValueError("MONGO_DB_URL environment variable is not set."), sys)


class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    
    def cv_to_json_convertor(self,file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
        
    def insert_data_mongodb(self,records,database,collection):
        try:
            # input validation
            if not isinstance(records, list):
                raise NetworkSecurityException(ValueError("records must be a list of dicts"), sys)
            if not isinstance(database, str) or not isinstance(collection, str):
                raise NetworkSecurityException(ValueError("database and collection must be strings"), sys)

            # assign attributes (no trailing commas)
            self.database = database
            self.collection = collection
            self.records = records

            # create client (use certifi CA bundle and short timeout)
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca, serverSelectionTimeoutMS=5000)

            # force DNS/connection check now
            self.mongo_client.admin.command("ping")

            # get database and collection
            db = self.mongo_client[self.database]
            coll = db[self.collection]

            result = coll.insert_many(self.records)
            return len(result.inserted_ids)
        except Exception as e:
            raise NetworkSecurityException(e,sys)      

if __name__ == '__main__':
    FILE_PATH="Network_Data\phisingData.csv"
    DATABASE="Network_db"
    Collection = "NetworkData"
    networkobj = NetworkDataExtract()
    records = networkobj.cv_to_json_convertor(file_path=FILE_PATH)
    print(records)
    no_of_records=networkobj.insert_data_mongodb(records,DATABASE,Collection)
    print(no_of_records)
