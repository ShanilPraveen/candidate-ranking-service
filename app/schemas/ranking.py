from pydantic import BaseModel

class RankingRequest(BaseModel):
    resumeID: str
    vacancyID: str

class RankingResponse(BaseModel):
    ranking_score: float 