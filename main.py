from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

model_path = os.path.join("model", "decision_tree_model.pkl")
model = joblib.load(model_path)

class CandidateFeatures(BaseModel):
    education_similarity: float
    experience_years: float
    cosine_similarity_skills: float
    highest_degree: int
    ed_req_encoded: int
    exp_req_encoded: int

@app.get("/")
def root():
    return {"message": "Candidate Ranking Model API is running!"}

@app.post("/predict")
def predict(data: CandidateFeatures):
    input_data = np.array([[
        data.education_similarity,
        data.experience_years,
        data.cosine_similarity_skills,
        data.highest_degree,
        data.ed_req_encoded,
        data.exp_req_encoded
    ]])
    prediction = model.predict(input_data)[0]
    return {"score": float(prediction)}
