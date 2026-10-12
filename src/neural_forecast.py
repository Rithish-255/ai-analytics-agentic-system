import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from src.config import CONFIG
from src.data_pipeline import prepare_features


def train_neural_model(df: pd.DataFrame):
    data = prepare_features(df)
    feature_cols = [
        "marketing_spend",
        "price_index",
        "weather_index",
        "promotions",
        "day_of_week",
        "month",
        "day_of_year",
        "lag_7",
        "lag_14",
        "rolling_mean_7",
        "rolling_mean_14",
    ]

    X = data[feature_cols].values
    y = data["sales"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=CONFIG.random_seed
    )

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        random_state=CONFIG.random_seed,
        max_iter=500,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)

    recent = data.iloc[-1:].copy()
    future_forecast = []
    for _ in range(CONFIG.prediction_days):
        values = recent.iloc[0].copy()
        sample = np.array(
            [[
                values["marketing_spend"],
                values["price_index"],
                values["weather_index"],
                values["promotions"],
                values["day_of_week"],
                values["month"],
                values["day_of_year"],
                values["lag_7"],
                values["lag_14"],
                values["rolling_mean_7"],
                values["rolling_mean_14"],
            ]]
        )
        prediction = float(model.predict(sample)[0])
        future_forecast.append(round(prediction, 2))
        recent["sales"] = prediction
        recent["lag_7"] = prediction
        recent["lag_14"] = prediction
        recent["rolling_mean_7"] = prediction
        recent["rolling_mean_14"] = prediction

    return {
        "model": model,
        "forecast": future_forecast,
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
        },
    }
