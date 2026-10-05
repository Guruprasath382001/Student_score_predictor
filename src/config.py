from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

MODEL_DIR = BASE_DIR / "models"

EXP_DIR = BASE_DIR / "experiments_results"

DATASET_PATH = DATA_DIR / "StudentPerformanceFactors.csv"

MODEL_PATH = MODEL_DIR / "student_score_pipeline.pkl"

METADATA_PATH = MODEL_DIR / "metadata.json"


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

CV_FOLDS = 5


# ============================================================
# TARGET
# ============================================================

TARGET = "Exam_Score"


# ============================================================
# FEATURES
# ============================================================

NUMERIC_FEATURES = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity",
]


CATEGORICAL_FEATURES = [
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Motivation_Level",
    "Internet_Access",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Peer_Influence",
    "Learning_Disabilities",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Gender",
]


ALL_FEATURES = (
    NUMERIC_FEATURES +
    CATEGORICAL_FEATURES
)


