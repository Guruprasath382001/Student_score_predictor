from fastapi import FastAPI
from experiments.tune_models import Tune_and_choose_right_model
from src.train import Train_model
from src.predict import predict_score
from experiments.outlier_analysis import outlier_analysis
from experiments.error_analysis import error_analysis
from experiments.model_comparison import model_comparison


# Create FastAPI app instance
app = FastAPI()


# Define a simple GET endpoint
@app.get("/")
def read_root():
    return {"message": "Hello World"}


@app.get("/choose_model/")
def choose_model():
    try:
        output = Tune_and_choose_right_model()

        return {
            "message":"success",
            "data":output
        }
    except Exception as e:
        return {
            "message":str(e)
        }


@app.get("/train/")
def Train():
    try:
        output = Train_model()

        return {
            "message":"success",
            "data":output
        }
    except Exception as e:
            return {
                "message":str(e)
            }


@app.post("/predict/")
def Predict(data:dict):
    try:
        output = predict_score(data)

        return {
            "message":"success",
            "data":output
        }
    except Exception as e:
            return {
                "message":str(e)
            }


@app.get("/analysis/")
def Analysis(type:str):
    try:

        if type=='outlier':
            outlier_analysis()
        elif type=='error':
            error_analysis()

        return {
            "message":"success",
        }
    
    except Exception as e:
            return {
                "message":str(e)
            }

@app.get("/model_comparison/")
def Model_comparison():
    try:

        model_comparison()

        return {
            "message":"success",
        }

    except Exception as e:
            return {
                "message":str(e)
            }