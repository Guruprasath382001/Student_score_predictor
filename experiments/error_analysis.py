import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import (
    ALL_FEATURES,
    DATASET_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TARGET,
)


def error_analysis():

    model = joblib.load(
        MODEL_PATH
    )

    df = pd.read_csv(
        DATASET_PATH
    )

    X = df[ALL_FEATURES]

    y = df[TARGET]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
        )
    )

    predictions = model.predict(
        X_test
    )

    results = X_test.copy()

    results["Actual"] = y_test.values

    results["Predicted"] = predictions

    results["Error"] = (
        results["Actual"] -
        results["Predicted"]
    )

    results["Absolute_Error"] = (
        results["Error"].abs()
    )

    print("=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    print(
        "\nMean Absolute Error:",
        results["Absolute_Error"].mean(),
    )

    print(
        "Maximum Absolute Error:",
        results["Absolute_Error"].max(),
    )

    print(
        "Minimum Absolute Error:",
        results["Absolute_Error"].min(),
    )

    print("\n" + "=" * 70)
    print("10 LARGEST PREDICTION ERRORS")
    print("=" * 70)

    columns = [
        "Hours_Studied",
        "Attendance",
        "Previous_Scores",
        "Actual",
        "Predicted",
        "Error",
        "Absolute_Error",
    ]

    print(
        results
        .sort_values(
            "Absolute_Error",
            ascending=False,
        )
        .head(10)[columns]
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("ERROR DISTRIBUTION")
    print("=" * 70)

    print(
        "Errors <= 1:",
        (
            results["Absolute_Error"] <= 1
        ).sum(),
    )

    print(
        "Errors <= 2:",
        (
            results["Absolute_Error"] <= 2
        ).sum(),
    )

    print(
        "Errors <= 3:",
        (
            results["Absolute_Error"] <= 3
        ).sum(),
    )

    print(
        "Errors > 3:",
        (
            results["Absolute_Error"] > 3
        ).sum(),
    )

    print(
        "Errors > 5:",
        (
            results["Absolute_Error"] > 5
        ).sum(),
    )

