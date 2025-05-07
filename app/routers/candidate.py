from fastapi import APIRouter, Depends
from ..schemas.candidate import CandidateFeatures
from ..services.prediction_service import PredictionService

router = APIRouter()

def get_prediction_service():
    return PredictionService()

@router.get("/")
def root():
    return {"message": "Candidate Ranking Model API is running!"}

@router.post("/predict")
def predict(
    data: CandidateFeatures,
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    score = prediction_service.predict_score(data)
    return {"score": score} 