import os
import pandas as pd
from joblib import load
from dotenv import load_dotenv
from pathlib import Path
import logging
load_dotenv()
ROOT_PATH=Path(os.getenv("PROJECT_ROOT"))
MODEL_PATH=ROOT_PATH/os.getenv("MODEL_DIR")/os.getenv("MODEL_NAME")
LOG_PATH=ROOT_PATH/os.getenv("LOG_DIR")/os.getenv("LOG_NAME")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(LOG_PATH)
            ]
        )
model=load(MODEL_PATH)
logging.info("model loaded successfully")
def predict(input_data: dict):
    df=pd.DataFrame([input_data])
    prediction=int(model.predict(df)[0])
    probability=float(model.predict_proba(df)[0][1])
    logging.info(f"Model provided a prediction: {prediction}, probability: {probability}")
    return {
        "prediction":prediction,
        "probability":probability
    }
    # example usage
# sample_input = {
#     "age": 52,
#     "sex": 1,
#     "cp": 0,
#     "trestbps": 125,
#     "chol": 212,
#     "fbs": 0,
#     "restecg": 1,
#     "thalach": 168,
#     "exang": 0,
#     "oldpeak": 1.0,
#     "slope": 2,
#     "ca": 0,
#     "thal": 2
# }
# result = predict(input_data=sample_input)
# print(result)
