import joblib
import pandas as pd

from src.config import (
    ALL_FEATURES,
    MODEL_PATH,
)


def load_model():
    """
    Load the trained ML pipeline.
    """

    return joblib.load(
        MODEL_PATH
    )


def validate_input(data):
    """
    Validate prediction input.
    """

    missing_features = [
        feature
        for feature in ALL_FEATURES
        if feature not in data
    ]

    if missing_features:

        raise ValueError(
            f"Missing features: "
            f"{missing_features}"
        )

    # Numeric validation
    if data["Hours_Studied"] < 0:
        raise ValueError(
            "Hours_Studied cannot be negative."
        )

    if not 0 <= data["Attendance"] <= 100:
        raise ValueError(
            "Attendance must be between 0 and 100."
        )

    if data["Sleep_Hours"] < 0:
        raise ValueError(
            "Sleep_Hours cannot be negative."
        )

    if not 0 <= data["Previous_Scores"] <= 100:
        raise ValueError(
            "Previous_Scores must be between 0 and 100."
        )

    if data["Tutoring_Sessions"] < 0:
        raise ValueError(
            "Tutoring_Sessions cannot be negative."
        )

    if data["Physical_Activity"] < 0:
        raise ValueError(
            "Physical_Activity cannot be negative."
        )


def predict_score(data):
    """
    Predict exam score for one student.
    """

    validate_input(data)

    input_df = pd.DataFrame(
        [data],
        columns=ALL_FEATURES,
    )

    model = load_model()

    prediction = model.predict(
        input_df
    )

    return float(
        prediction[0]
    )



    # student = {
    #     "Hours_Studied": 2,
    #     "Attendance": 80,
    #     "Parental_Involvement": "Low",
    #     "Access_to_Resources": "Medium",
    #     "Extracurricular_Activities": "No",
    #     "Sleep_Hours": 9,
    #     "Previous_Scores": 80,
    #     "Motivation_Level": "Low",
    #     "Internet_Access": "No",
    #     "Tutoring_Sessions": 2,
    #     "Family_Income": "Low",
    #     "Teacher_Quality": "Low",
    #     "School_Type": "Private",
    #     "Peer_Influence": "Neutral",
    #     "Physical_Activity": 1,
    #     "Learning_Disabilities": "Yes",
    #     "Parental_Education_Level": "High School",
    #     "Distance_from_Home": "Moderate",
    #     "Gender": "Male",
    # }

    # score = predict_score(
    #     student
    # )

    # print(
    #     f"Predicted Exam Score: {score:.2f}"
    # )