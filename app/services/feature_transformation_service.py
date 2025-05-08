from typing import List, Dict, Any, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class FeatureTransformationService:
    # Education degree mapping dictionaries
    EDUCATION_ALIASES = {
        "phd": ["doctor of philosophy", "ph.d", "ph.d.", "phd", "doctorate", "ph.d. in", "phd candidate"],
        "mba": ["master of business administration", "mba executive", "executive mba", "mba", "masters of business administration"],
        "msc": ["master of science", "m.sc", "m.s", "masters of science", "msc", "masters in science", "m.sc."],
        "ma": ["master of arts", "m.a", "m.a.", "masters of arts"],
        "mcom": ["master of commerce", "m.com", "mcom"],
        "me": ["master of engineering", "m.e", "m.eng", "m.e.", "m.engg"],
        "mtech": ["master of technology", "m.tech", "mtech", "mtech integrated"],
        "bsc": ["bachelor of science", "b.sc", "b.s", "bsc", "b.sc.", "b.s.", "honours bachelor of science", "bachelors of science"],
        "ba": ["bachelor of arts", "b.a", "ba", "b.a.", "bachelors of arts"],
        "bcom": ["bachelor of commerce", "b.com", "bcom"],
        "be": ["bachelor of engineering", "b.e", "b.e.", "b.eng", "b.engg", "bachelor of engineering (b.e"],
        "btech": ["bachelor of technology", "b.tech", "b.tech.", "btech", "b.tech(computers)", "dual degree (b.tech + m.tech)", "integrated b.tech & m.tech"],
        "bba": ["bachelor of business administration", "b.b.a", "bba", "bba - accounting", "bba - finance", "bachelor business administration"],
        "bca": ["bachelor of computer applications", "b.c.a", "bca"],
        "mca": ["master of computer applications", "m.c.a", "mca"],
        "bs": ["bs", "b.s", "b.s.", "b.s in", "bachelor's degree in science", "bachelor's in science"],
        "ms": ["ms", "m.s", "m.s.", "master in computer science", "masters of science in information technology"],
        "aa": ["associate of arts", "a.a", "aa"],
        "aas": ["associate of applied science", "a.a.s", "aas"],
        "as": ["associate of science", "a.s", "as", "associate of science degree"],
        "associate": ["associate's degree", "associate degree", "associates degree", "associates", "associate"],
        "diploma": ["technical diploma", "associate diploma", "polytechnic diploma", "diploma", "general diploma", "pg diploma", "master's diploma"],
        "high school": ["high school diploma", "ged", "grade 12", "xii", "x", "kcse"],
        "certificate": ["certificate of completion", "graduate certificate", "business certification", "epa certification", "aws brazing certification", "skills", "course", "certification", "minor", "training", "coaching"],
        "others": ["n/a", "select one", "attending", "testing computer software", "general courses"],
        "al": ["advanced level", "a/l", "a.l", "gce a/l", "gce advanced level", "gce (a/l)", "gce(al)", "gce-a/l"],
        "ol": ["ordinary level", "o/l", "o.l", "gce o/l", "gce ordinary level", "gce (o/l)", "gce(ol)", "gce-o/l"],
        "nvq": ["nvq", "nvq level 3", "nvq level 4", "nvq level 5", "nvq level 6", "national vocational qualification", "nvq diploma"],
        "hnd": ["hnd", "higher national diploma", "hnd in", "higher national diploma in"],
        "cima": ["cima", "chartered institute of management accountants", "cima qualification"],
        "acca": ["acca", "association of chartered certified accountants"],
        "ca": ["chartered accountant", "institute of chartered accountants of sri lanka", "ica", "ca sri lanka"],
        "slim": ["slim", "slim diploma", "sri lanka institute of marketing", "slim pgd"],
        "nibt": ["nibt", "national institute of business & technology", "nibt diploma"],
        "bit": ["bit", "bachelor of information technology", "bit degree", "bit (colombo university)"]
    }

    EDUCATION_RANKS = {
        "others": 0,
        "high school": 1,
        "certificate": 1,
        "ol": 1,
        "al": 2,
        "diploma": 3,
        "associate": 3,
        "nvq": 3,
        "hnd": 3,
        "aa": 3,
        "aas": 3,
        "as": 3,
        "slim": 3,
        "nibt": 3,
        "bsc": 4,
        "bs": 4,
        "ba": 4,
        "be": 4,
        "btech": 4,
        "bit": 4,
        "cima": 4,
        "acca": 4,
        "bcom": 4,
        "bba": 4,
        "bca": 4,
        "msc": 5,
        "ms": 5,
        "ma": 5,
        "me": 5,
        "mtech": 5,
        "mcom": 5,
        "mba": 5,
        "mca": 5,
        "ca": 5,
        "phd": 6
    }

    # Flattened aliases for easy reverse lookup
    FLATTENED_ALIASES = {}
    for canonical, synonyms in EDUCATION_ALIASES.items():
        for synonym in synonyms:
            FLATTENED_ALIASES[synonym.lower()] = canonical

    @staticmethod
    def get_highest_education(degree_entries):
        if not isinstance(degree_entries, list):
            degree_entries = [degree_entries]

        best_match = ("unknown", -1)

        for entry in degree_entries:
            if not isinstance(entry, str):
                continue

            # Clean and normalize text
            text = re.sub(r'[^\w\s]', '', entry.lower())

            # Try matching from aliases
            for synonym, canonical in FeatureTransformationService.FLATTENED_ALIASES.items():
                if synonym in text:
                    rank = FeatureTransformationService.EDUCATION_RANKS.get(canonical, -1)
                    if rank > best_match[1]:
                        best_match = (canonical, rank)

            # Fallback to direct canonical keyword matching
            for canonical, rank in FeatureTransformationService.EDUCATION_RANKS.items():
                if canonical in text:
                    if rank > best_match[1]:
                        best_match = (canonical, rank)

        return best_match[1]

    @staticmethod
    def normalize_education_text(text: str) -> str:
        replacements = {
            r'\bb[\s\.-]*sc\b': 'bachelor of science',
            r'\bm[\s\.-]*sc\b': 'master of science',
            r'\bb[\s\.-]*tech\b': 'bachelor of technology',
            r'\bm[\s\.-]*tech\b': 'master of technology',
            r'\bb[\s\.-]*e\b': 'bachelor of engineering',
            r'\bm[\s\.-]*e\b': 'master of engineering',
            r'\bb[\s\.-]*a\b': 'bachelor of arts',
            r'\bm[\s\.-]*a\b': 'master of arts',
            r'\bb[\s\.-]*com\b': 'bachelor of commerce',
            r'\bm[\s\.-]*com\b': 'master of commerce',
            r'\bb[\s\.-]*ba\b': 'bachelor of business administration',
            r'\bm[\s\.-]*ba\b': 'master of business administration',
            r'\bmba\b': 'master of business administration',
            r'\bbba\b': 'bachelor of business administration',
            r'\bph[\s\.-]*d\b': 'doctor of philosophy',
            r'\bd[\s\.-]*phil\b': 'doctor of philosophy',
            r'\bbachelor/honors\b': 'bachelor degree',
            r'\bdiploma\b': 'diploma',
            r'\bmasters?\b': 'master degree',
        }

        text = text.lower()
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        return text
    
    EDUCATION_RANKS = {
    "others": 0,
    "high school": 1,
    "certificate": 1,
    "ol": 1,
    "al": 2,
    "diploma": 3,
    "associate": 3,
    "nvq": 3,
    "hnd": 3,
    "aa": 3,
    "aas": 3,
    "as": 3,
    "slim": 3,
    "nibt": 3,
    "bachelor of science": 4,
    "bachelor of arts": 4,
    "bachelor of engineering": 4,
    "bachelor of technology": 4,
    "bachelor of commerce": 4,
    "bachelor of business administration": 4,
    "bachelor degree": 4,
    "bit": 4,
    "bca": 4,
    "bcom": 4,
    "cima": 4,
    "acca": 4,
    "master of science": 5,
    "master of arts": 5,
    "master of engineering": 5,
    "master of technology": 5,
    "master of commerce": 5,
    "master of business administration": 5,
    "master degree": 5,
    "mca": 5,
    "ca": 5,
    "doctor of philosophy": 6,
    "phd": 6,
    "doctorate": 6,
    "philosophy doctor": 6,}

    @staticmethod
    def encode_required_education(text: Union[str, List[str]]) -> int:
        if isinstance(text, list):
            text = " ".join(text)

        text = FeatureTransformationService.normalize_education_text(text)

        best_rank = 0
        for key, rank in FeatureTransformationService.EDUCATION_RANKS.items():
            if key in text:
                best_rank = max(best_rank, rank)
        return best_rank



    @staticmethod
    def compute_education_similarity(candidate_degree: List[str], candidate_major: List[str], required_degree: List[str], required_major: List[str]) -> float:
        # Join fields into strings
        candidate_text = " ".join(candidate_degree + candidate_major)
        required_text = " ".join(required_degree + required_major)

        # Normalize and lowercase
        candidate_text = FeatureTransformationService.normalize_education_text(candidate_text)
        required_text = FeatureTransformationService.normalize_education_text(required_text)

        # Handle empty inputs safely
        if not candidate_text.strip() or not required_text.strip():
            return 0.0

        # Compute TF-IDF vectors
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([candidate_text, required_text])

        # Compute cosine similarity
        similarity_matrix = cosine_similarity(vectors[0:1], vectors[1:2])
        return similarity_matrix[0][0]
        


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
        features['education_similarity'] = FeatureTransformationService.compute_education_similarity(
            record.get('higest_degree_name',[]),
            record.get('major_field_of_study',[]),
            record.get('required_education_degree_name',[]),
            record.get('required_education_major_field_of_study',[])
        )

        print("Education similarity:", features['education_similarity'])

        features['experience_years'] = record.get('experience', 0)

        print("Experience years:", features['experience_years'])

        features['cosine_similarity_skills'] = FeatureTransformationService.compute_skill_similarity(
            record.get('skills', []),
            record.get('required_skills', [])
        )

        print("Cosine similarity skills:", features['cosine_similarity_skills'])

        features['highest_degree'] = FeatureTransformationService.encode_required_education(
            record.get('higest_degree_name', []),

        )

        print("Highest degree:", features['highest_degree'])

        features['ed_req_encoded'] = FeatureTransformationService.encode_required_education(
            record.get('required_education_degree_name', []),

        )

        print("Education required encoded:", features['ed_req_encoded'])

        features['exp_req_encoded'] = record.get('required_experience', 0)

        print("Experience required encoded:", features['exp_req_encoded'])

  
        
        # Add experience_years feature directly from experience field
        features['experience_years'] = record.get('experience', 0)
        
        # Add exp_req_encoded feature directly from required_experience field
        features['exp_req_encoded'] = record.get('required_experience', 0)
        
        # Transform highest degree into encoded value
        features['highest_degree'] = FeatureTransformationService.get_highest_education(
            record.get('higest_degree_name', '')
        )
        
        return features 