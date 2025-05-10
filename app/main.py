from fastapi import FastAPI, HTTPException, Depends
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from typing import Dict, Any, List, Union
import pandas as pd
import joblib
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import urllib.parse
from contextlib import asynccontextmanager
from bson import ObjectId
from datetime import datetime

# MongoDB connection settings
MONGODB_URL = f"mongodb+srv://ashidudissanayake1:chP0CyGcYR89zDeg@cluster0.4dg71cd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DATABASE_NAME = "resume_rover_db"

# Async client for async operations
client = AsyncIOMotorClient(MONGODB_URL)
db = client[DATABASE_NAME]

# Load the model once when the module is loaded
MODEL_PATH = Path("decision_tree_model.joblib")
final_model = joblib.load(MODEL_PATH)

# Threshold for triggering bias analysis after N candidate predictions per job
BIAS_TRIGGER_THRESHOLD = 100
BIAS_CHECK_SAMPLE_SIZE = 100
BIAS_THRESHOLD = 0.15

# Degree mapping and splitting functions
DEGREE_MAP = {
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

# Create a reverse mapping from degree representations to standardized abbreviations
DEGREE_VARIANTS = {}
for abbrev, variants in DEGREE_MAP.items():
    for variant in variants:
        DEGREE_VARIANTS[variant.lower()] = abbrev.upper()

# Sort variants by length in descending order to match longer variants first
SORTED_VARIANTS = sorted(DEGREE_VARIANTS.keys(), key=len, reverse=True)

# List of connector words to remove from the beginning of the field
CONNECTOR_WORDS = ['in', 'of', 'on', 'for', 'at', 'with', 'to', 'and']

# List of degree qualifiers that should be part of the degree name
DEGREE_QUALIFIERS = [
    r'eng\(hons\)', r'eng\s*\(hons\)',  # Eng(Hons)
    r'\(hons\)', r'\(honors\)',  # (Hons), (Honors)
    r'with\s*honors', r'with\s*honours',  # with honors/honours
    r'with\s*distinction',  # with distinction
    r'first\s*class',  # first class
    r'second\s*class',  # second class
    r'upper\s*division',  # upper division
    r'lower\s*division',  # lower division
]

def split_degree_field(text):
    """
    Splits a degree string into standardized degree abbreviation and field of study.

    Parameters:
        text (str): The input string containing degree information.

    Returns:
        tuple: (degree_abbreviation (str or None), field_of_study (str))
    """
    text_lower = text.lower()
    degree_abbrev = None
    field = text
    
    # First, try to find a degree match
    for variant in SORTED_VARIANTS:
        pattern = r'\b' + re.escape(variant) + r'\b'
        match = re.search(pattern, text_lower)
        if match:
            degree_abbrev = DEGREE_VARIANTS[variant]
            # Extract the field by removing the matched degree variant from the original text
            field = re.sub(pattern, '', text, flags=re.IGNORECASE).strip(" ,:-")
            break
    
    # If no standard degree found, check for G.C.E
    if not degree_abbrev:
        if "g.c.e" in text_lower or "gce" in text_lower:
            if "a/l" in text_lower or "advanced level" in text_lower:
                degree_abbrev = "AL"
                # Remove all G.C.E related terms and clean the field
                field = text.replace("G.C.E", "").replace("A/L", "").replace("Advanced Level", "").replace("GCE", "").strip(" ,:-")
                # If field is empty or only contains G.C.E related terms, set it to empty
                if not field or all(term in text_lower for term in ["g.c.e", "gce", "a/l", "advanced level"]):
                    field = ""
            elif "o/l" in text_lower or "ordinary level" in text_lower:
                degree_abbrev = "OL"
                # Remove all G.C.E related terms and clean the field
                field = text.replace("G.C.E", "").replace("O/L", "").replace("Ordinary Level", "").replace("GCE", "").strip(" ,:-")
                # If field is empty or only contains G.C.E related terms, set it to empty
                if not field or all(term in text_lower for term in ["g.c.e", "gce", "o/l", "ordinary level"]):
                    field = ""
    
    if degree_abbrev:
        # Look for degree qualifiers after the degree abbreviation
        qualifier_pattern = '|'.join(DEGREE_QUALIFIERS)
        qualifier_match = re.search(qualifier_pattern, field.lower())
        
        if qualifier_match:
            # Extract the qualifier and add it to the degree
            qualifier = field[qualifier_match.start():qualifier_match.end()]
            degree_abbrev = f"{degree_abbrev} {qualifier}"
            # Remove the qualifier from the field
            field = re.sub(qualifier_pattern, '', field, flags=re.IGNORECASE).strip(" ,:-")
        
        # Handle comma-separated fields
        if ',' in field:
            # Take everything after the first comma as the field
            field = field.split(',', 1)[1].strip()
        
        # Remove leading connector words from the field
        field_tokens = field.split()
        while field_tokens and field_tokens[0].lower() in CONNECTOR_WORDS:
            field_tokens.pop(0)
        field = ' '.join(field_tokens)
        
        return degree_abbrev, field
    
    # If no degree variant is matched, return None for degree and the original text as field
    return None, text.strip()



def calculate_total_experience(work_experience: list) -> float:
    total_months = 0

    for exp in work_experience:
        # 1. Handle duration field
        duration = exp.get("duration", "")
        if duration:
            duration = duration.lower()
            months = 0

            # Extract years and months
            year_match = re.search(r"(\d+)\s*year", duration)
            month_match = re.search(r"(\d+)\s*month", duration)

            if year_match:
                months += int(year_match.group(1)) * 12
            if month_match:
                months += int(month_match.group(1))

            total_months += months
            continue

        # 2. Handle dates field
        dates = exp.get("dates", "")
        if not dates:
            continue

        # Normalize dashes
        dates = re.sub(r"[–—−]", "-", dates)

        try:
            start_str, end_str = [s.strip() for s in dates.split("-")]
        except ValueError:
            continue  # Invalid format

        # Handle "Present"
        if "present" in end_str.lower():
            end_date = datetime.today() # Current date assumed
        else:
            try:
                end_date = datetime.strptime(end_str, "%B %Y")  # Full month
            except ValueError:
                try:
                    end_date = datetime.strptime(end_str, "%b %Y")  # Abbr month
                except:
                    continue

        try:
            start_date = datetime.strptime(start_str, "%B %Y")
        except ValueError:
            try:
                start_date = datetime.strptime(start_str, "%b %Y")
            except:
                continue

        months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
        total_months += max(0, months)

    return round(total_months / 12, 2)

# Pydantic Models
class CandidateFeatures(BaseModel):
    education_similarity: float
    experience_years: float
    cosine_similarity_skills: float
    highest_degree: int
    ed_req_encoded: int
    exp_req_encoded: int

class RankingRequest(BaseModel):
    resumeID: str

class RankingResponse(BaseModel):
    ranking_score: float

# Database Functions
async def get_database():
    return db

async def initialize_database():
    try:
        await client.admin.command('ping')
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        raise e

@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialize_database()
    print("Connected to MongoDB!")
    yield
    client.close()
    print("Disconnected from MongoDB!")

# Feature Transformation Service
class FeatureTransformationService:
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
    def compute_skill_similarity(candidate_skills: List[str], required_skills: List[str]) -> float:
        if not candidate_skills or not required_skills:
            return 0.0

        candidate_skills_str = " ".join(candidate_skills)
        required_skills_str = " ".join(required_skills)

        if not candidate_skills_str or not required_skills_str:
            return 0.0

        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([candidate_skills_str, required_skills_str])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error computing similarity: {str(e)}")
            return 0.0
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
    def transform_record_to_features(record: Dict[str, Any]) -> Dict[str, Any]:
        features = {}
        
        features['education_similarity'] = FeatureTransformationService.compute_education_similarity(
            record.get('higest_degree_name',[]),
            record.get('major_field_of_study',[]),
            record.get('required_education_degree_name',[]),
            record.get('required_education_major_field_of_study',[])
        )

        features['experience_years'] = record.get('experience', 0)

        features['cosine_similarity_skills'] = FeatureTransformationService.compute_skill_similarity(
            record.get('skills', []),
            record.get('required_skills', [])
        )

        features['highest_degree'] = FeatureTransformationService.encode_required_education(
            record.get('higest_degree_name', [])
        )

        features['ed_req_encoded'] = FeatureTransformationService.encode_required_education(
            record.get('required_education_degree_name', [])
        )

        features['exp_req_encoded'] = record.get('required_experience', 0)
        
        return features

# Bias Analysis and Mitigation
def check_bias(records: List[Dict]) -> List[int]:
    if len(records) < BIAS_CHECK_SAMPLE_SIZE:
        print(f"Not enough records for bias check. Records count: {len(records)}")
        return []

    group_scores = {}
    for record in records:
        group = record.get("highest_degree")
        score = record.get("ranking_score")
        if group is not None and score is not None:
            group_scores.setdefault(group, []).append(score)

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
    if not biased_groups:
        return records

    scores = [r["ranking_score"] for r in records if r["ranking_score"] is not None]
    global_mean = np.mean(scores)
    
    print(f"Global Mean for Reweighing: {global_mean}")

    for record in records:
        if record.get("highest_degree") in biased_groups:
            current_score = record.get("ranking_score", 0)
            adjusted_score = (current_score + global_mean) / 2
            print(f"Adjusting score for record with degree {record['highest_degree']} from {current_score} to {adjusted_score}")
            record["ranking_score"] = round(float(adjusted_score), 4)

    return records

# Ranking Service
class RankingService:
    @staticmethod
    async def calculate_and_update_ranking(resumeID: str) -> float:
        db = await get_database()

        resume_obj_id = ObjectId(resumeID)
        resume_data = await db.parsed_resumes.find_one({
            "_id": resume_obj_id
        })
        if not resume_data:
            raise ValueError(f"No resume found for resumeID {resumeID}")
        
        # print("Resume Data:", resume_data)
        
        job_obj_id = ObjectId(resume_data.get("job_id"))
        job_data = await db.jobs.find_one({
            "_id": job_obj_id
        })
        if not job_data:
            raise ValueError(f"No job found for jobID {resume_data.get('job_id')}")

        # print("Job Data:", job_data)
        
        # Find the highest degree from all education entries
        highest_degree = None
        highest_degree_text = None
        highest_rank = -1

        for edu in resume_data["education"]:
            degree, major = split_degree_field(edu["degree"])
            if degree:
                # Get the base degree without qualifiers for ranking
                base_degree = degree.split()[0] if degree else None
                if base_degree:
                    # Find the rank of this degree in EDUCATION_RANKS
                    for key, rank in FeatureTransformationService.EDUCATION_RANKS.items():
                        if key.upper() == base_degree:
                            if rank > highest_rank:
                                highest_rank = rank
                                highest_degree = degree  # Keep the full degree with qualifiers
                                highest_degree_text = edu["degree"]
                            break

        # If no degree was found, use the first education entry
        if highest_degree is None and resume_data["education"]:
            highest_degree, major = split_degree_field(resume_data["education"][0]["degree"])
            highest_degree_text = resume_data["education"][0]["degree"]
        else:
            # Get the major field from the highest degree text
            _, major = split_degree_field(highest_degree_text)
        
        # Split required education into degree and major field
        required_degree, required_major = split_degree_field(job_data["required_education"])

        total_experience = calculate_total_experience(resume_data["work_experience"])
        
        pre_record = {
            "higest_degree_name": [highest_degree] if highest_degree else [],  # Highest degree found
            "major_field_of_study": [major] if major else [],  # Major field of highest degree
            "experience": total_experience,  # Will calculate from work_experience dates
            "skills": resume_data["skills"]["technical_skills"],
            "required_education_degree_name": [required_degree] if required_degree else [],  # Required degree
            "required_education_major_field_of_study": [required_major] if required_major else [],  # Required major field
            "required_experience": job_data["required_experience"],
            "required_skills": job_data["required_skills"]
        }

        # Calculate total experience in years from work_experience dates
        # total_experience = 0
        # for exp in resume_data["work_experience"]:
        #     dates = exp["dates"].split("–")
        #     if len(dates) == 2:
        #         start_date = dates[0].strip()
        #         end_date = dates[1].strip()
                
        #         # Simple calculation assuming each month is 1/12 of a year
        #         # This is a basic implementation - you might want to use a more sophisticated date parsing
        #         if "Present" in end_date:
        #             end_date = "2025"  # Use current year as end date
                
        #         try:
        #             start_year = int(start_date.split()[-1])
        #             end_year = int(end_date.split()[-1])
        #             total_experience += (end_year - start_year)
        #         except (ValueError, IndexError):
        #             continue

        print("Pre Record:", pre_record)
        record = pre_record
        
        features = FeatureTransformationService.transform_record_to_features(record)
        features_df = pd.DataFrame([features])
        ranking_score = final_model.predict(features_df)[0]
        print("Predicted ranking score:", ranking_score)

        await db.parsed_resumes.update_one(
            {"_id": resume_obj_id},
            {"$set": {"ranking_score": ranking_score}}
        )

        # await db.bias_tracking.update_one(
        #     {"_id": resume_obj_id},
        #     {"$inc": {"count": 1}},
        #     upsert=True
        # )

        # bias_counter_doc = await db.bias_tracking.find_one({"vacancyID": vacancyID})
        # if bias_counter_doc and bias_counter_doc.get("count", 0) >= BIAS_TRIGGER_THRESHOLD:
        #     print(f"[INFO] Triggering bias check for vacancyID: {vacancyID}")
        #     records = await db.match_data.find({"vacancyID": vacancyID}).to_list(length=None)
        #     biased_groups = check_bias(records)
        #     if biased_groups:
        #         updated_records = apply_reweighing(records, biased_groups)
        #         for record in updated_records:
        #             await db.match_data.update_one(
        #                 {"_id": record["_id"]},
        #                 {"$set": {"ranking_score": record["ranking_score"]}}
        #             )

        #     await db.bias_tracking.update_one(
        #         {"vacancyID": vacancyID},
        #         {"$set": {"count": 0}}
        #     )

        return ranking_score

# Prediction Service
class PredictionService:
    def __init__(self):
        self.model = final_model
    
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

# FastAPI App Setup
app = FastAPI(lifespan=lifespan)

def get_prediction_service():
    return PredictionService()

@app.get("/")
def root():
    return {"message": "Candidate Ranking Model API is running!"}

@app.post("/predict")
def predict(
    data: CandidateFeatures,
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    score = prediction_service.predict_score(data)
    return {"score": score}

@app.post("/ranking", response_model=RankingResponse)
async def calculate_ranking(request: RankingRequest):
    try:
        ranking_score = await RankingService.calculate_and_update_ranking(
            request.resumeID
        )
        return RankingResponse(ranking_score=ranking_score)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
