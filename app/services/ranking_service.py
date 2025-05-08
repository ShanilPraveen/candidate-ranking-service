from typing import Dict, Any
import pandas as pd
import joblib
from pathlib import Path

from app.database import get_database
from app.services.feature_transformation_service import FeatureTransformationService

# Load the model once when the module is loaded
MODEL_PATH = Path("decision_tree_model.joblib")
final_model = joblib.load(MODEL_PATH)

class RankingService:
    @staticmethod
    async def calculate_and_update_ranking(resumeID: str, vacancyID: str) -> float:
        """
        Calculate ranking score for a candidate and update it in the database
        """
        # Get database connection
        db = await get_database()
        
        # Find the record
        record = await db.match_data.find_one({
            "resumeID": resumeID,
            "vacancyID": vacancyID
        })
        
        if not record:
            raise ValueError(f"No record found for resumeID {resumeID} and vacancyID {vacancyID}")
        
        # Transform record to features
        features = FeatureTransformationService.transform_record_to_features(record)
        
        # Convert feature dict to a single-row DataFrame
        features_df = pd.DataFrame([features])

        # Predict ranking score using the trained model
        ranking_score = final_model.predict(features_df)[0]
        print("Predicted ranking score:", ranking_score)
        
        # Update the record in the database
        await db.match_data.update_one(
            {"resumeID": resumeID, "vacancyID": vacancyID},
            {"$set": {"ranking_score": ranking_score}}
        )
        
        return ranking_score 