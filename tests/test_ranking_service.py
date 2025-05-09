import pytest
from unittest import mock
from app.services.ranking_service import RankingService

@pytest.mark.asyncio
async def test_calculate_and_update_ranking():
    # Mock input data
    resumeID = "test_resume_id"
    vacancyID = "test_vacancy_id"

    # Mock database record
    mock_record = {
        "resumeID": resumeID,
        "vacancyID": vacancyID,
        "other_data": "mock_data"  # add other mock fields as needed
    }

    # Mock the database call to return our mock record
    with mock.patch("app.services.ranking_service.get_database") as mock_db:
        # Create a mock db object with a find_one method that returns the mock_record
        mock_db.return_value.match_data.find_one.return_value = mock_record

        # Mock the database call for bias_tracking.find_one to return a mock document
        mock_db.return_value.bias_tracking.find_one.return_value = {"count": 50}  # Set mock count

        # Mock the model prediction to return a fixed score
        with mock.patch("app.services.ranking_service.final_model.predict") as mock_predict:
            mock_predict.return_value = [0.85]  # The predicted ranking score

            # Call the method (we'll mock the database and model predictions in the test)
            score = await RankingService.calculate_and_update_ranking(resumeID, vacancyID)

            # Assert that the returned score is the mocked prediction score
            assert score == 0.85

            # Assert that the count in the bias_tracking collection was checked
            mock_db.return_value.bias_tracking.find_one.assert_called_once_with({"vacancyID": vacancyID})

            # Assert that the update for the bias_tracking count was triggered
            mock_db.return_value.bias_tracking.update_one.assert_called_once_with(
                {"vacancyID": vacancyID},
                {"$inc": {"count": 1}},
                upsert=True
            )

            # Assert that the record was updated with the predicted score
            mock_db.return_value.match_data.update_one.assert_called_once_with(
                {"resumeID": resumeID, "vacancyID": vacancyID},
                {"$set": {"ranking_score": 0.85}}
            )
