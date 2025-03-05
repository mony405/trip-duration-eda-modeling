# Import libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from apply_featurengineering import apply_feature_engineering
from feature_engineering import remove_outliers
from loaddata import *
from evaluate import *
from preprocessing import *
import pickle
import warnings
warnings.filterwarnings('ignore')

if __name__=="__main__":
    # load train val test
    df_train,df_val,df_test=load_data()

    # feature engineering
    df_train, df_val, df_test=apply_feature_engineering(df_train,df_val,df_test)

    # divide categorical and numerical columns
    categorical_cols = ['passenger_count', 'store_and_fwd_flag','day_of_week', 'hour', 'month', 'day_of_year']

    num_cols = ['pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude'
        ,'manhattan_distance','euclidean_distance', 'haversine_distance', 'bearing','bearing_cos','bearing_sin']

    # remove outliers
    df_train, df_val, df_test=remove_outliers(df_train, df_val, df_test, num_cols)

    # split data to X and y
    X_train, y_train, X_val, y_val, X_test, y_test=split_xy(df_train,df_val,df_test)

    # preprocessing
    column_transformer=preprocessing(categorical_cols, num_cols)

    user_choice=int(input("1-train mode\n2-inference mode\nEnter your choice : "))
    saving_path = 'models/Best_approach_model.pkl'
    if user_choice==1:
        # Model Training
        pipeline = Pipeline(steps=[
            ('ohe', column_transformer),
            ('regression', Ridge(alpha=1))
        ])
        train_feat = categorical_cols + num_cols
        model = pipeline.fit(X_train[train_feat], y_train)

        with open(saving_path, 'wb') as file:
            pickle.dump(model, file)
        print("saved successfully")
        evaluate(X_train, y_train, model, 'train')
        evaluate(X_val, y_val, model, 'val')

    elif user_choice==2 :
        with open(saving_path, 'rb') as file:
            model = pickle.load(file)
        # evaluate the model
        evaluate(X_test, y_test, model, 'test')

    else:
        print("Invalid choice or model is not trained yet")

