from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FeatureTransformationService:
    @staticmethod
    def clean_skills(skills: List[str]) -> List[str]:
        """
        Clean and normalize a list of skills
        """
        if not skills:
            return []
        
        # Clean each skill
        cleaned_skills = []
        for skill in skills:
            # Remove bullet points and other unwanted characters
            skill = skill.replace('•', '').strip().lower()
            if skill:  # Only add non-empty skills
                cleaned_skills.append(skill)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_skills = [x for x in cleaned_skills if not (x in seen or seen.add(x))]
        
        return unique_skills

    @staticmethod
    def compute_skill_similarity(candidate_skills: List[str], required_skills: List[str]) -> float:
        """
        Compute cosine similarity between candidate skills and required skills
        """
        # Clean both skill lists
        cleaned_candidate_skills = FeatureTransformationService.clean_skills(candidate_skills)
        cleaned_required_skills = FeatureTransformationService.clean_skills(required_skills)
        
        # Convert lists to space-separated strings
        candidate_skills_str = " ".join(cleaned_candidate_skills)
        required_skills_str = " ".join(cleaned_required_skills)

        # If either string is empty, return 0 similarity
        if not candidate_skills_str or not required_skills_str:
            return 0.0

        # Prepare and fit a TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        try:
            # Transform the skill strings into TF-IDF vectors
            tfidf_matrix = vectorizer.fit_transform([candidate_skills_str, required_skills_str])
            
            # Compute cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error computing similarity: {str(e)}")
            return 0.0

    @staticmethod
    def transform_record_to_features(record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform a database record into features expected by the model
        """
        features = {}
        
        # Transform skills into cosine similarity feature
        features['cosine_similarity_skills'] = FeatureTransformationService.compute_skill_similarity(
            record.get('skills', []),
            record.get('required_skills', [])
        )
        
        # Add other feature transformations here as needed
        # For example:
        # features['education_match'] = ...
        # features['experience_ratio'] = ...
        
        return features 