from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,MinMaxScaler,OneHotEncoder,PolynomialFeatures
from sklearn.pipeline import Pipeline

def preprocessing(categorical_cols,num_cols):
    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        # One-hot encoding for categorical
        ('poly', Pipeline([  # Apply PolynomialFeatures + Scaling to numerical
            ('poly_features', PolynomialFeatures(degree=3, include_bias=False)),
            ('scaling', StandardScaler())
        ]), num_cols)
    ], remainder='passthrough')

    return column_transformer




