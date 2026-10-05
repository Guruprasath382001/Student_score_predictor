import pytest

from src.predict import predict_score


VALID_STUDENT = {
    "Hours_Studied": 2,
    "Attendance": 80,
    "Parental_Involvement": "Low",
    "Access_to_Resources": "Medium",
    "Extracurricular_Activities": "No",
    "Sleep_Hours": 9,
    "Previous_Scores": 80,
    "Motivation_Level": "Low",
    "Internet_Access": "No",
    "Tutoring_Sessions": 2,
    "Family_Income": "Low",
    "Teacher_Quality": "Low",
    "School_Type": "Private",
    "Peer_Influence": "Neutral",
    "Physical_Activity": 1,
    "Learning_Disabilities": "Yes",
    "Parental_Education_Level": "High School",
    "Distance_from_Home": "Moderate",
    "Gender": "Male",
}


def test_prediction_returns_number():

    score = predict_score(
        VALID_STUDENT
    )

    assert isinstance(
        score,
        float,
    )


def test_prediction_is_reasonable():

    score = predict_score(
        VALID_STUDENT
    )

    assert 0 <= score <= 100


def test_negative_hours_are_rejected():

    student = VALID_STUDENT.copy()

    student["Hours_Studied"] = -1

    with pytest.raises(ValueError):

        predict_score(
            student
        )


def test_invalid_attendance_is_rejected():

    student = VALID_STUDENT.copy()

    student["Attendance"] = 150

    with pytest.raises(ValueError):

        predict_score(
            student
        )


def test_negative_tutoring_sessions_are_rejected():

    student = VALID_STUDENT.copy()

    student["Tutoring_Sessions"] = -1

    with pytest.raises(ValueError):

        predict_score(
            student
        )