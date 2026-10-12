import numpy as np
import pandas as pd

from src.config import CONFIG


def generate_synthetic_sales_data(days: int = 365) -> pd.DataFrame:
    """Generate a realistic synthetic sales dataset with trend, seasonality, and noise."""
    dates = pd.date_range(start="2023-01-01", periods=days, freq="D")
    time_index = np.arange(days)

    trend = 1200 + 0.9 * time_index
    seasonal = 180 * np.sin(2 * np.pi * time_index / 30.5)
    weekly = 90 * np.sin(2 * np.pi * time_index / 7)
    marketing_effect = 200 * np.sin(2 * np.pi * time_index / 21)
    noise = np.random.default_rng(CONFIG.random_seed).normal(0, 35, size=days)

    sales = trend + seasonal + weekly + marketing_effect + noise
    sales = np.clip(sales, 400, None)

    df = pd.DataFrame(
        {
            "date": dates,
            "sales": sales.round(2),
            "marketing_spend": (np.abs(np.sin(2 * np.pi * time_index / 17)) * 180 + 60).round(2),
            "price_index": (1.1 + 0.12 * np.sin(2 * np.pi * time_index / 25)).round(4),
            "weather_index": (60 + 25 * np.sin(2 * np.pi * time_index / 18)).round(2),
            "promotions": (np.random.default_rng(CONFIG.random_seed + 1).uniform(0, 1, size=days) > 0.7).astype(int),
        }
    )

    return df


def prepare_features(df: pd.DataFrame):
    """Create engineered features for supervised learning and forecasting tasks."""
    data = df.copy()
    data["day_of_week"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month
    data["day_of_year"] = data["date"].dt.dayofyear
    data["lag_7"] = data["sales"].shift(7).fillna(method="bfill")
    data["lag_14"] = data["sales"].shift(14).fillna(method="bfill")
    data["rolling_mean_7"] = data["sales"].shift(1).rolling(window=7, min_periods=1).mean()
    data["rolling_mean_14"] = data["sales"].shift(1).rolling(window=14, min_periods=1).mean()
    return data.dropna().reset_index(drop=True)
