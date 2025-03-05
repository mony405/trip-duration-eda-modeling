import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder

# Constants to Convert Latitude/Longitude to Kilometers
LAT_TO_KM = 111  # 1 degree latitude ≈ 111 km
LON_TO_KM = 111  # 1 degree longitude ≈ 111 km at the equator


def drop_unnecessary_columns(train, val, test, column_name):
    train = train.drop(columns=[column_name], axis=1)
    val = val.drop(columns=[column_name], axis=1)
    test = test.drop(columns=[column_name], axis=1)
    return train, val, test


def transform_datatype(df_train, df_val, df_test, column_name, column_type):
    df_train[column_name] = column_type(df_train[column_name])
    df_val[column_name] = column_type(df_val[column_name])
    df_test[column_name] = column_type(df_test[column_name])
    return df_train, df_val, df_test


def datetime_columns(df_train, df_val, df_test, date_col, columns_names, columns_type):
    for name, dtype in zip(columns_names, columns_type):
        df_train[name] = getattr(df_train[date_col].dt, dtype)
        df_val[name] = getattr(df_val[date_col].dt, dtype)
        df_test[name] = getattr(df_test[date_col].dt, dtype)
    return df_train, df_val, df_test


def generate_bearing_col(df_train, df_val, df_test):
    for df in [df_train, df_val, df_test]:
        df["bearing"] = \
            calculate_bearing(df["pickup_latitude"], df["pickup_longitude"], df["dropoff_latitude"],
                              df["dropoff_longitude"])
        df["bearing_sin"] = np.sin(np.radians(df["bearing"]))
        df["bearing_cos"] = np.cos(np.radians(df["bearing"]))

    return df_train, df_val, df_test

def calculate_bearing(lat1, lon1, lat2, lon2):
    delta_lon = np.radians(lon2 - lon1)
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    x = np.sin(delta_lon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)

    return np.degrees(np.arctan2(x, y))

def remove_outliers(df_train, df_val, df_test, num_colss):
    cleaned_dfs = []
    num_cols=num_colss.copy()
    num_cols.append("trip_duration")
    for df in [df_train, df_val, df_test]:
        # Compute Z-scores for numerical columns
        z_scores = df[num_cols].apply(zscore)

        # Define a threshold (e.g., |z| > 3 is considered an outlier)
        threshold = 3

        # Create a boolean mask where True indicates an outlier
        outlier_mask = (z_scores.abs() > threshold).any(axis=1)

        # Remove outliers and reset index
        df_cleaned = df[~outlier_mask].reset_index(drop=True)

        cleaned_dfs.append(df_cleaned)

    return cleaned_dfs  # Returns the cleaned versions of df_train, df_val, df_test


def apply_log(df_train, df_val, df_test, col_name):
    """Apply log transformation to a column in all datasets."""
    for df in [df_train, df_val, df_test]:
        df[col_name] = np.log1p(df[col_name])  # log(1 + x) to avoid log(0)
    return df_train, df_val, df_test

def haversine_distance(lat1, lon1, lat2, lon2):
    """Compute Haversine distance between two latitude/longitude points in kilometers."""
    R = 6371  # Radius of Earth in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])  # Convert degrees to radians
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def add_distance_columns(df_train, df_val, df_test):
    """Compute Manhattan, Euclidean, and Haversine distances (all in KM)."""
    for df in [df_train, df_val, df_test]:
        # Convert latitude & longitude differences to KM
        delta_lat_km = (df["dropoff_latitude"] - df["pickup_latitude"]) * LAT_TO_KM
        delta_lon_km = (df["dropoff_longitude"] - df["pickup_longitude"]) * LON_TO_KM

        # Manhattan Distance (sum of absolute differences)
        df["manhattan_distance"] = abs(delta_lat_km) + abs(delta_lon_km)

        # Euclidean Distance (straight-line distance)
        df["euclidean_distance"] = np.sqrt(delta_lat_km**2 + delta_lon_km**2)

        # Haversine Distance (more accurate for real-world travel)
        df['haversine_distance'] = df.apply(
            lambda row: haversine_distance(row['pickup_latitude'], row['pickup_longitude'],
                                           row['dropoff_latitude'], row['dropoff_longitude']), axis=1)

    return df_train, df_val, df_test


# def day_periods(df_train, df_val, df_test):
#     def categorize_hour(hour):
#         if 6 <= hour < 10:
#             return "Morning Rush"
#         elif 10 <= hour < 15:
#             return "Midday"
#         elif 15 <= hour < 19:
#             return "Evening Rush"
#         elif 19 <= hour < 23:
#             return "Night"
#         else:
#             return "Late Night"
#     for df in [df_train, df_val, df_test]:
#         df["hour_category"] = df["hour"].apply(categorize_hour)
#         # df.drop(["hour"], axis=1, inplace=True)

# def weekend(df_train, df_val, df_test):
#     for df in [df_train, df_val, df_test]:
#         df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
#         # df.drop(["day_of_week"], axis=1, inplace=True)
#     return df_train, df_val, df_test

# def holidays(df_train, df_val, df_test):
#     def categorize_month(month):
#         if month in [12, 1, 2]:
#             return "Winter"
#         elif month in [3, 4, 5]:
#             return "Spring"
#         elif month in [6, 7, 8]:
#             return "Summer"
#         else:
#             return "Fall"
#
#     for df in [df_train, df_val, df_test]:
#         df["season"] = df["month"].apply(categorize_month)
#
#     return df_train, df_val, df_test
#
# def holiday(df_train, df_val, df_test):
#     public_holidays = ["2024-01-01", "2024-07-04", "2024-12-25"]  # Example for US holidays
#     for df in [df_train, df_val, df_test]:
#         df["is_holiday"] = df["pickup_datetime"].dt.strftime("%Y-%m-%d").isin(public_holidays).astype(int)
#         # df.drop(["day_of_year"], axis=1, inplace=True)
#         # df.drop(["pickup_datetime"], axis=1, inplace=True)
#
#     return df_train, df_val, df_test
