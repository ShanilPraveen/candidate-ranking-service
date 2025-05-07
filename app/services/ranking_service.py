from typing import Dict, Any
from app.database import get_database
from app.services.feature_transformation_service import FeatureTransformationService

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
        
        # TODO: Add your model prediction here
        # For now, we'll just use the cosine similarity as the ranking score
        ranking_score = features['cosine_similarity_skills']
        
        # Update the record in the database
        await db.match_data.update_one(
            {"resumeID": resumeID, "vacancyID": vacancyID},
            {"$set": {"ranking_score": ranking_score}}
        )
        
        return ranking_score 