import joblib
import numpy as np
import os
from ..schemas.candidate import CandidateFeatures

class PredictionService:
    def __init__(self):
        model_path = os.path.join("decision_tree_model.joblib")
        self.model = joblib.load(model_path)
    
    def predict_score(self, data: CandidateFeatures) -> float:
        input_data = np.array([[
            data.education_similarity,
            data.experience_years,
            data.cosine_similarity_skills,
            data.highest_degree,
            data.ed_req_encoded,
            data.exp_req_encoded
        ]])
        prediction = self.model.predict(input_data)[0]
        return float(prediction) 