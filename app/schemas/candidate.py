from pydantic import BaseModel

class CandidateFeatures(BaseModel):
    education_similarity: float
    experience_years: float
    cosine_similarity_skills: float
    highest_degree: int
    ed_req_encoded: int
    exp_req_encoded: int 