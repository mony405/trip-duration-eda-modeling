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




'''
without poly
Train R2 Score:  0.3525586724315113
Train MSE:  0.4089349023889545
Validation R2 Score:  0.3846393775051622
Validation MSE:  0.39385332898108744
Test R2 Score:  0.41041368771618425
Test MSE:  0.37348022316082047
'''
'''
bearing column
Train R2 Score:  0.3416010161849794
Train MSE:  0.4158559435656367
Validation R2 Score:  0.3894645029564652
Validation MSE:  0.3907650720269094
Test R2 Score:  0.4079547056890741
Test MSE:  0.37503789357666867
'''
'''
rush hour only
Train R2 Score:  0.33287443069330613
Train MSE:  0.4213678026859502
Validation R2 Score:  0.3633680565132148
Validation MSE:  0.4074677532361494
Test R2 Score:  0.3877642786189228
Test MSE:  0.38782775156821986
'''
'''
log the distance
Train R2 Score:  0.6259958247912325
Train MSE:  0.23622736821025675
Validation R2 Score:  0.6260531719642792
Validation MSE:  0.2393396614925948

Train R2 Score:  0.6403108629470171
Train MSE:  0.22718574778587822
Validation R2 Score:  0.6399509889201549
Validation MSE:  0.23044454979134624

Train R2 Score:  0.6425579594753646
Train MSE:  0.22576644357968953
Validation R2 Score:  0.6423958318194808
Validation MSE:  0.22887976082121128

Train R2 Score:  0.6469328546140303
Train MSE:  0.22300318575181652
Validation R2 Score:  0.6474212449429458
Validation MSE:  0.22566331242359133

Train R2 Score:  0.6469328546139462
Train MSE:  0.2230031857518696
Validation R2 Score:  0.6474212449428741
Validation MSE:  0.2256633124236372

Train R2 Score:  0.6469328546140303
Train MSE:  0.22300318575181652
Validation R2 Score:  0.6474212449429458
Validation MSE:  0.22566331242359133

Train R2 Score:  0.657242500150417
Train MSE:  0.16695869965251228
Validation R2 Score:  0.6546449515305733
Validation MSE:  0.16572139635974176

Train R2 Score:  0.6535956694366921
Train MSE:  0.1687350870811858
Validation R2 Score:  0.6507519335829437
Validation MSE:  0.16758949231836168

Train R2 Score: 0.6795
Validation R2 Score: 0.6812

Train R2 Score: 0.6721
Train MSE:  0.15671863179368523
Validation R2 Score: 0.6744
Validation MSE:  0.15572939945157732

Train R2 Score: 0.6791
Train MSE:  0.15597690643710782
Validation R2 Score: 0.6773
Validation MSE:  0.15486176140909688

Train R2 Score: 0.6967
Train MSE:  0.14744052669066215
Validation R2 Score: 0.6938
Validation MSE:  0.14691173507195238

Train R2 Score: 0.6967
Train MSE:  0.14744052347087563
Validation R2 Score: 0.6938
Validation MSE:  0.14691174612290475

bearing sin/cos
train R2 Score: 0.7091
train MSE: , 0.1414172838216291
val R2 Score: 0.7063
val MSE: , 0.14095043305133237

'''