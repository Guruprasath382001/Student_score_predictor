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


def outlier_analysis():

    df = pd.read_csv(
        DATASET_PATH
    )

    model = joblib.load(
        MODEL_PATH
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

    results["Actual_Score"] = (
        y_test.values
    )

    results["Predicted_Score"] = (
        predictions
    )

    results["Absolute_Error"] = (
        results["Actual_Score"] -
        results["Predicted_Score"]
    ).abs()


    high_error = results

    # high_error = results[                     ### uncomment this if you want only error greater than 5
    #     results["Absolute_Error"] > 5
    # ]


    print("=" * 80)
    print("HIGH-ERROR STUDENTS")
    print("=" * 80)

    # print(
    #     high_error[
    #         ALL_FEATURES +
    #         [
    #             "Actual_Score",
    #             "Predicted_Score",
    #             "Absolute_Error",
    #         ]
    #     ].to_string(index=False)
    # )

    
    high_error_df =  high_error[
                ALL_FEATURES +
                [
                    "Actual_Score",
                    "Predicted_Score",
                    "Absolute_Error",
                ]
            ]

    # Save to Excel
    excel_filename = "experiments_results\high_pricing_errors.xlsx"
    high_error_df.to_excel(excel_filename, index=False)

    print(f" Saved high errors to Excel sheet: {excel_filename}")


    # Save to HTML Webpage
    html_filename = "experiments_results\high_pricing_errors.html"
    high_error_df.to_html(html_filename, index=False, classes='table table-striped')

    print(f" Saved high errors to HTML report: {html_filename}")
        

    # print(
    #     "\nNumber of high-error students:",
    #     len(high_error),
    # )
