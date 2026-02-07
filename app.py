import sys
import os 

import certifi 
ca  = certifi.where()

from dotenv import load_dotenv
from Networksecurity.Exception.exception import NetworkSecurityException
from Networksecurity.Logging.logger import logging

load_dotenv()

mongo_db_url = os.getenv("MONGO_DB_URL")
print(mongo_db_url)

import pymongo
from Networksecurity.pipeline.training_pipeline import TrainingPipeline
from Networksecurity.utils.ml_utils.model.estimater import NetworkModel
from fastapi.middleware.cors import CORSMiddleware
from  fastapi import FastAPI , File , UploadFile , Request
from uvicorn import run as app_run 
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd 
from Networksecurity.utils.main_utils.utils import save_numpy_array_data,load_numpy_array_data
from Networksecurity.utils.main_utils.utils import save_object,load_object


client = pymongo.MongoClient(mongo_db_url,tlsCAFile=ca)

app = FastAPI()
origins = ["*"]  # Allow all origins for simplicity, adjust as needed

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")


@app.get("/" , tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train():
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
        return Response(content="Training pipeline executed successfully", media_type="text/plain")
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise NetworkSecurityException(e, sys)
    
@app.post("/predict")
async def predict_route(request:Request,file: UploadFile = File(...)):
    try:
       df=pd.read_csv(file.file)
       
       # print(df)
       preprocessor = load_object(file_path="final_model/preprocessor.pkl")
       final_model = load_object(file_path="final_model/model.pkl")
       network_model = NetworkModel(preprocessor=preprocessor, model=final_model)
       print(df.iloc[0])
       y_pred = network_model.predict(df.drop(columns=["Result"], errors="ignore"))
       print(y_pred)
       df["predicted_column"] = y_pred
       print(df["predicted_column"])
       # df["predicted_column"].replace(-1,0)
       # return df.to_json()
       df.to_csv("prediction_output/output.csv",index=False)
       table_html = df.to_html(classes="table table-striped")
       # print(table_html)
       return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
    except Exception as e:
        
        
        logging.error(f"Error during prediction: {e}")
        raise NetworkSecurityException(e, sys)
    



    
if __name__ == "__main__":app_run(app, host="localhost", port=8000)