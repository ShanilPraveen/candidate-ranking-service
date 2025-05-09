from app.services import bias

def test_check_bias_detects_biased_group():
    # Simulated records with biased group (group '1' has much higher scores)
    records = [
        {"highest_degree": 1, "ranking_score": 90},
        {"highest_degree": 1, "ranking_score": 92},
        {"highest_degree": 2, "ranking_score": 60},
        {"highest_degree": 2, "ranking_score": 62},
    ] * 25  # Repeat to reach 100 items

    biased_groups = bias.check_bias(records)
    assert 1 in biased_groups or 2 in biased_groups

def test_apply_reweighing_adjusts_scores():
    records = [
        {"highest_degree": 1, "ranking_score": 90},
        {"highest_degree": 1, "ranking_score": 92},
        {"highest_degree": 2, "ranking_score": 60},
        {"highest_degree": 2, "ranking_score": 62},
    ] * 25

    biased_groups = bias.check_bias(records)
    adjusted_records = bias.apply_reweighing(records, biased_groups)

    for r in adjusted_records:
        assert "ranking_score" in r
        assert isinstance(r["ranking_score"], float)
