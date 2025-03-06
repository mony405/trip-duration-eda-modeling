# NYC Taxi Trip Duration Prediction

This project aims to predict the duration of taxi trips in New York City using machine learning techniques. A Ridge regression model was trained on a **sample of the dataset**, incorporating advanced feature engineering and preprocessing techniques to improve performance.

## Dataset
The dataset is from Kaggle's [NYC Taxi Trip Duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/overview) competition. It contains details of taxi trips, including pickup and dropoff locations, timestamps, and other relevant information.

## Feature Engineering
Feature engineering was applied to extract meaningful insights from raw data. Two versions of feature engineering were implemented: **Baseline** and **Best Model**.

### Baseline Feature Engineering
- Dropped unnecessary columns (`id`, `vendor_id`).
- Converted `pickup_datetime` to a datetime object.
- Extracted time-based features (`day_of_week`, `hour`, `month`, `day_of_year`).

### Best Model Feature Engineering
- Included all baseline feature engineering steps.
- Added distance-based features: `haversine_distance`, `manhattan_distance`, and `euclidean_distance`.
- Applied logarithmic transformation to distance columns.
- Computed `bearing` (direction of travel).
- Applied log transformation to the target variable (`trip_duration`).

## Preprocessing
### Baseline Preprocessing
- Applied **One-Hot Encoding** for categorical features.
- Standardized numerical features using **StandardScaler**.

### Best Model Preprocessing
- Applied **One-Hot Encoding** for categorical features.
- Applied **PolynomialFeatures (degree=3) + StandardScaler** for numerical features.

## Model Training
The model pipeline consists of:
1. Feature Engineering
2. Data Preprocessing
3. Training a **Ridge Regression model** (`alpha=1`)

## Performance Comparison
| Model        | Train R²  | Train MSE  | Validation R²  | Validation MSE  |
|-------------|-----------|-----------|----------------|-----------------|
| Baseline    | 0.3326    | 0.4089    | 0.3246         | 0.3939          |
| Best Model  | 0.7091    | 0.1414    | 0.7063         | 0.1409          |

The **Best Model** significantly outperforms the **Baseline Model**, showing a much higher R² score and lower Mean Squared Error (MSE) on both training and validation sets.

## Notes
- The model was trained on a **sample** of the dataset rather than the entire dataset due to computational constraints.
- Some additional features (e.g., public holidays, seasons) were tested but did not significantly improve model performance, so they were excluded.

## How to Run
1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Run the training script:
   ```bash
   python main.py
   
