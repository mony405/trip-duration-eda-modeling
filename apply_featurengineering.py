from feature_engineering import *


def apply_feature_engineering(df_train,df_val,df_test):
    # drop id , vendor id
    df_train, df_val, df_test = \
        drop_unnecessary_columns(df_train, df_val, df_test, 'vendor_id')

    df_train, df_val, df_test = \
        drop_unnecessary_columns(df_train, df_val, df_test, 'id')

    # transform 'pickup_datetime' to date to extract important columns
    df_train, df_val, df_test = \
        transform_datatype(df_train, df_val, df_test, 'pickup_datetime', pd.to_datetime)

    # make new columns from 'pickup_datetime' after transforming
    columns_names = ["day_of_week", "hour", "month", "day_of_year"]
    columns_type = ["dayofweek", "hour", "month", "dayofyear"]

    df_train, df_val, df_test = \
        datetime_columns(df_train, df_val, df_test, 'pickup_datetime', columns_names, columns_type)

    # generate new column "haversine_distance", "manhattan_distance", "euclidean_distance" from latitude and longitude
    df_train, df_val, df_test = add_distance_columns(df_train, df_val, df_test)

    # Apply log transformation to distance columns
    for col in ["haversine_distance", "manhattan_distance", "euclidean_distance"]:
        df_train, df_val, df_test = apply_log(df_train, df_val, df_test, col)

    # generate new column called bearing from latitude and longitude,bearing angle represents the direction of travel
    df_train, df_val, df_test = generate_bearing_col(df_train, df_val, df_test)

    # log the trip duration column , because it has big values
    df_train, df_val, df_test = apply_log(df_train, df_val, df_test, 'trip_duration')
    '''Didn't affect model's accuracy , so decided not to invest in these features'''
    # # Convert Hours into Categories (Traffic Congestion Levels)
    # df_train, df_val, df_test = day_periods(df_train, df_val, df_test)
    # # Group Days of the Week (Weekday vs. Weekend)
    # df_train, df_val, df_test = weekend(df_train, df_val, df_test)
    # # Group Months (Seasons)
    # df_train, df_val, df_test = holidays(df_train, df_val, df_test)
    # # Mark Public Holidays
    # df_train, df_val, df_test = holiday(df_train, df_val, df_test)

    return df_train,df_val,df_test

