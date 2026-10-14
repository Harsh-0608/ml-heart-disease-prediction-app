from fastapi import FastAPI
from pydantic import BaseModel
from backend.prediction import predict
app=FastAPI(title="Heart disease prediction API",
            version="1.0.0")
class HeartDiseaseInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int
@app.get("/health")
def health_check():
    return {"status":"ok"}
@app.post("/predict-heart-disease")
def predict_heart_disease(input_data:HeartDiseaseInput):
    input_dict=input_data.model_dump()
    result=predict(input_data=input_dict)
    return {
        "prediction":result["prediction"],
        "probability":result["probability"],
        "diagnosis":(
            "Heart disease is detected"
            if result["prediction"]==1
            else "No Heart Disease Detected"
        )
    }

