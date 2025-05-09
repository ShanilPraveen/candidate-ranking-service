from typing import Dict, Any
import pandas as pd
import joblib
from pathlib import Path
import os

from app.database import get_database
from app.services.feature_transformation_service import FeatureTransformationService
from app.services import bias  # Import bias analysis and mitigation module

# Load the model once when the module is loaded
MODEL_PATH = Path("D:\semester 4\SE\project tasks\candidate-ranking-service\decision_tree_model.joblib")
final_model = joblib.load(MODEL_PATH)

# Threshold for triggering bias analysis after N candidate predictions per job
BIAS_TRIGGER_THRESHOLD = 100

class RankingService:
    @staticmethod
    async def calculate_and_update_ranking(resumeID: str, vacancyID: str) -> float:
        """
        Calculate ranking score for a candidate, update it in the database,
        and periodically check and mitigate bias based on the number of scored CVs.

        Steps:
        - Retrieve candidate-job match record from DB
        - Extract features from the record
        - Predict ranking score using pre-trained model
        - Update the predicted score in the database
        - Track number of scored CVs per job in 'bias_tracking' collection
        - Trigger bias check + mitigation every BIAS_TRIGGER_THRESHOLD (e.g. 100) CVs
        """
        # Get DB connection
        db = await get_database()
        
        # Step 1: Find the matching record
        record = await db.match_data.find_one({
            "resumeID": resumeID,
            "vacancyID": vacancyID
        })
        if not record:
            raise ValueError(f"No record found for resumeID {resumeID} and vacancyID {vacancyID}")
        
        # Step 2: Extract structured features using transformation service
        features = FeatureTransformationService.transform_record_to_features(record)

        # Step 3: Convert to DataFrame format suitable for prediction
        features_df = pd.DataFrame([features])

        # Step 4: Predict the ranking score
        ranking_score = final_model.predict(features_df)[0]
        print("Predicted ranking score:", ranking_score)

        # Step 5: Update the record with the predicted score
        await db.match_data.update_one(
            {"resumeID": resumeID, "vacancyID": vacancyID},
            {"$set": {"ranking_score": ranking_score}}
        )

        # Step 6: Track scoring frequency in 'bias_tracking' collection
        await db.bias_tracking.update_one(
            {"vacancyID": vacancyID},
            {"$inc": {"count": 1}},
            upsert=True
        )

        # Step 7: Check if it's time to run bias analysis + mitigation
        bias_counter_doc = await db.bias_tracking.find_one({"vacancyID": vacancyID})
        if bias_counter_doc and bias_counter_doc.get("count", 0) >= BIAS_TRIGGER_THRESHOLD:
            print(f"[INFO] Triggering bias check for vacancyID: {vacancyID}")
            await bias.check_and_mitigate_for_vacancy(vacancyID)

            # Reset the count after mitigation
            await db.bias_tracking.update_one(
                {"vacancyID": vacancyID},
                {"$set": {"count": 0}}
            )

        return ranking_score
