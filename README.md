# NYC Trip Duration Prediction - Ridge Regression

## Overview
This project focuses on predicting the duration of NYC taxi trips using machine learning. The Ridge regression model was trained with feature engineering and preprocessing techniques to improve performance over a baseline model.

Dataset: [NYC Taxi Trip Duration (Kaggle)](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/overview)

## Feature Engineering
### Baseline Model
The baseline model includes basic feature extraction from the pickup timestamp:
- **Datetime Features**: Extracted `day_of_week`, `hour`, `month`, and `day_of_year` from `pickup_datetime`.
- **Dropped Unnecessary Columns**: Removed `id` and `vendor_id`.

### Best Model
In addition to the baseline features, the best model includes advanced feature engineering:
- **Distance Features**: `haversine_distance`, `manhattan_distance`, and `euclidean_distance` were computed using latitude and longitude.
- **Bearing Feature**: Represents the direction of travel.
- **Log Transformation**: Applied to `trip_duration` and distance-related features to reduce skewness.
- **Additional Features**: Tested various holiday and traffic congestion-related features, but they were excluded as they did not improve accuracy.

## Preprocessing
### Baseline Model
- **Categorical Features**: One-hot encoded.
- **Numerical Features**: Standardized using `StandardScaler`.

### Best Model
- **Categorical Features**: One-hot encoded.
- **Numerical Features**: Applied **PolynomialFeatures (degree=3)** to create interactions, followed by `StandardScaler`.

## Model Training
Both models were trained using a Ridge regression model with `alpha=1`.

## Performance Comparison
| Model        | Train R²  | Train MSE  | Validation R²  | Validation MSE  |
|-------------|----------|------------|---------------|----------------|
| Baseline    | 0.3326   | 0.4089     | 0.3246        | 0.3939         |
| Best Model  | 0.7091   | 0.1414     | 0.7063        | 0.1409         |

The best model significantly outperformed the baseline, showing higher R² and lower MSE on both training and validation sets.

## How to Run the Project
1. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
2. Run the main script:  
   ```bash
   python main.py
   ```
3. Choose between training (`1`) or inference (`2`).

## Conclusion
By incorporating advanced feature engineering and preprocessing techniques, the Ridge model's performance improved significantly over the baseline. The project showcases the importance of feature selection and transformation in predictive modeling.
