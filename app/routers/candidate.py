from fastapi import APIRouter, Depends, HTTPException
from ..schemas.candidate import CandidateFeatures
from ..services.prediction_service import PredictionService
from app.schemas.ranking import RankingRequest, RankingResponse
from app.services.ranking_service import RankingService

router = APIRouter(
    prefix="/candidates",
    tags=["candidates"]
)

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

@router.post("/ranking", response_model=RankingResponse)
async def calculate_ranking(request: RankingRequest):
    try:
        ranking_score = await RankingService.calculate_and_update_ranking(
            request.resumeID,
            request.vacancyID
        )
        return RankingResponse(ranking_score=ranking_score)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 