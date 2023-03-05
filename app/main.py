#-----------------------------------------------------------------------------------------------------------------------------------
__author__ = "Christian Simonis"
__copyright__ = "Copyright 2023"
__version__ = "1.0.0"
__maintainer__ = "Christian Simonis"
__email__ = "christian.Simonis.1989@gmail.com"
__status__ = "work in progress"

#-----------------------------------------------------------------------------------------------------------------------------------
from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import execute_mdl
from app.model.model import __version__ as mdl_version
#-----------------------------------------------------------------------------------------------------------------------------------


# API
app = FastAPI()

# Input class
class DataIn(BaseModel):
    age: float 
    sex: float 
    bmi: float 
    bp: float 
    s1: float 
    s2: float 
    s3: float 
    s4: float 
    s5: float
    s6: float 

#Output Class
class PredictionOut(BaseModel):
    mean: float
    var: float




# Getting
@app.get("/")
def home():
    return {"mdl_version": mdl_version}

#Posting
@app.post("/predict", response_model=PredictionOut)
def predict(payload: DataIn):
 
# Features:
    age=  payload.age 
    sex=  payload.sex 
    bmi=  payload.bmi 
    bp =  payload.bp   
    s1 =  payload.s1   
    s2 =  payload.s2   
    s3 =  payload.s3   
    s4 =  payload.s4  
    s5 =  payload.s5  
    s6 =  payload.s6   

    # Model Execution
    mean, var = execute_mdl([[age,sex,bmi,bp,s1,s2,s3,s4,s5,s6]])
    print('____________________________________________________________')
    print(mean)
    print(type(mean))
    print('____________________________________________________________')
    return {"mean": float(mean),
            "var": float(var)
            }


""" # Reponse body
{
  "age": 0.1,
  "sex": 0.2,
  "bmi": 0.3,
  "bp": 0.4,
  "s1": 0.5,
  "s2": 0.6,
  "s3": 0.7,
  "s4": 0.8,
  "s5": 0.9,
  "s6": 0.99
}
"""