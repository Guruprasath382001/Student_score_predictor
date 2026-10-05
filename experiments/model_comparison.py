import pandas as pd

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.config import (
    ALL_FEATURES,
    CV_FOLDS,
    DATASET_PATH,
    RANDOM_STATE,
    TARGET,
)

from src.preprocessing import create_preprocessor
import os,json


def model_comparison():

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

    models = {
        "Linear Regression": (
            LinearRegression(),
            {}
        ),


        "Ridge": (
            Ridge(),
            {
                "model__alpha": [
                    0.01,
                    0.1,
                    1,
                    10,
                    100,
                ]
            },
        ),

        "Random Forest": (
            RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "model__n_estimators": [
                    100,
                    200,
                ],
                "model__max_depth": [
                    5,
                    10,
                    None,
                ],
                "model__min_samples_split": [
                    2,
                    5,
                ],
            },
        ),

        "Gradient Boosting": (
            GradientBoostingRegressor(
                random_state=RANDOM_STATE,
            ),
            {
                "model__n_estimators": [
                    100,
                    200,
                ],
                "model__learning_rate": [
                    0.05,
                    0.1,
                ],
                "model__max_depth": [
                    2,
                    3,
                    5,
                ],
            },
        ),
    }

    results = []

    for name, (model, params) in models.items():

        print(f"\nTuning {name}...")

        pipeline = Pipeline([
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "model",
                model,
            ),
        ])

        search = GridSearchCV(
            pipeline,
            params,
            cv=CV_FOLDS,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )

        search.fit(
            X_train,
            y_train,
        )

        predictions = search.predict(
            X_test
        )

        mse = mean_squared_error(
            y_test,
            predictions,
        )

        rmse = mse ** 0.5

        mae = mean_absolute_error(
            y_test,
            predictions,
        )

        r2 = r2_score(
            y_test,
            predictions,
        )

        results.append({
            "Model": name,
            "CV_RMSE": -search.best_score_,
            "Test_MSE": mse,
            "Test_RMSE": rmse,
            "Test_MAE": mae,
            "Test_R2": r2,
            "Best_Parameters": search.best_params_,
        })

    
    RESULTS_PATH = (
        "experiments_results/model_comparison_results.json"
    )

    with open(
            RESULTS_PATH,
            "w",
            encoding="utf-8"
        ) as file:
    
        json.dump(
            results,
            file,
            indent=4
        )


        print("\nResults saved to:")
    
        print(
            RESULTS_PATH
        )

