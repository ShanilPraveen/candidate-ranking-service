from typing import List, Dict
import numpy as np

BIAS_CHECK_SAMPLE_SIZE = 100
BIAS_THRESHOLD = 0.15  # Adjust as needed (e.g., 15% deviation in mean scores)

def check_bias(records: List[Dict]) -> List[int]:
    """
    Check if bias exists based on score distribution across highest_degree groups.
    Returns list of biased group IDs (if any).
    """
    if len(records) < BIAS_CHECK_SAMPLE_SIZE:
        print(f"Not enough records for bias check. Records count: {len(records)}")
        return []

    group_scores = {}
    for record in records:
        group = record.get("highest_degree")
        score = record.get("ranking_score")
        if group is not None and score is not None:
            group_scores.setdefault(group, []).append(score)

    # Compute mean scores per group
    means = {k: np.mean(v) for k, v in group_scores.items()}
    global_mean = np.mean([score for scores in group_scores.values() for score in scores])

    biased_groups = []
    for group, mean in means.items():
        deviation = abs(mean - global_mean) / global_mean
        print(f"Group {group} - Mean: {mean}, Global Mean: {global_mean}, Deviation: {deviation}")
        if deviation > BIAS_THRESHOLD:
            biased_groups.append(group)

    return biased_groups


def apply_reweighing(records: List[Dict], biased_groups: List[int]) -> List[Dict]:
    """
    Apply simple reweighing mitigation to records belonging to biased groups.
    """
    if not biased_groups:
        return records  # No changes if no bias found

    # Compute global mean
    scores = [r["ranking_score"] for r in records if r["ranking_score"] is not None]
    global_mean = np.mean(scores)
    
    print(f"Global Mean for Reweighing: {global_mean}")

    for record in records:
        if record.get("highest_degree") in biased_groups:
            current_score = record.get("ranking_score", 0)
            adjusted_score = (current_score + global_mean) / 2  # Simple averaging strategy
            print(f"Adjusting score for record with degree {record['highest_degree']} from {current_score} to {adjusted_score}")
            record["ranking_score"] = round(float(adjusted_score), 4)

    return records
