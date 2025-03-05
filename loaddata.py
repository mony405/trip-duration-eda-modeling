import pandas as pd

def load_data():
    df_train= pd.read_csv('data/train.csv')
    df_val = pd.read_csv('data/val.csv')
    df_test = pd.read_csv('data/test.csv')
    return df_train, df_val, df_test

def split_xy(df_train,df_val,df_test):
    y_train=df_train['trip_duration']
    X_train=df_train.drop(columns=['trip_duration'])
    y_val=df_val['trip_duration']
    X_val=df_val.drop(columns=['trip_duration'])
    y_test=df_test['trip_duration']
    X_test=df_test.drop(columns=['trip_duration'])
    return X_train,y_train,X_val,y_val,X_test,y_test
