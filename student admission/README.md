# Graduate Admission Prediction

## Dataset Preview
```
First five rows:
<copy-paste output here>

Data info:
<copy-paste output here>

Missing values per column:
<copy-paste output here>
```

## Data Analysis

### Correlation Matrix
![Correlation Matrix](correlation_matrix.png)

### CGPA Distribution
![CGPA Histogram](cgpa_histogram.png)

## Model Evaluation

```
Mean Squared Error: <your MSE here>
R2 Score: <your R2 here>
```

### Feature Importances
![Feature Importance](feature_importance.png)

## Sample Predictions

```
Predicted Chance of Admission: 0.93 (for GRE=330, TOEFL=115, ...)
Predicted Chance of Admission: <second prediction here>
```

## How to Run

1. Clone this repo.
2. Install requirements:
   ```
   pip install pandas scikit-learn matplotlib seaborn joblib
   ```
3. Place `Admission_Predict.csv` in the repo folder.
4. Run `admission_prediction_full.py`