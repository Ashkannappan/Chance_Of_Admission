import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Dataset
data = pd.read_csv('Admission_Predict.csv')

# 2. Explore Data
print("First five rows:")
print(data.head())
print("\nData info:")
print(data.info())
print("\nMissing values per column:")
print(data.isnull().sum())

# 3. Data Visualization
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.show()

plt.figure(figsize=(6,4))
data['CGPA'].hist(bins=20)
plt.xlabel('CGPA')
plt.ylabel('Count')
plt.title('CGPA Distribution')
plt.savefig('cgpa_histogram.png')
plt.show()

# 4. Prepare Features and Target (use exact column names!)
X = data.drop(['Chance of Admit ', 'Serial No.'], axis=1)
y = data['Chance of Admit ']

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.4f}')
print(f'R2 Score: {r2:.4f}')

# 8. Feature Importance Visualization
features = X.columns
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8,6))
sns.barplot(x=importances[indices], y=features[indices])
plt.title('Feature Importances')
plt.savefig('feature_importance.png')
plt.show()

# 9. Predict Admission Chance for a New Applicant
def predict_admission(gre, toefl, university_rating, sop, lor, cgpa, research):
    input_data = pd.DataFrame([[gre, toefl, university_rating, sop, lor, cgpa, research]],
                              columns=X.columns)
    prediction = model.predict(input_data)[0]
    print(f'Predicted Chance of Admission: {prediction:.2f}')
    return prediction

# Example prediction
predict_admission(
    gre=330,
    toefl=115,
    university_rating=5,
    sop=5,
    lor=5,
    cgpa=9.5,
    research=1
)

# 10. Save Model for Future Use
joblib.dump(model, 'admission_model.pkl')
print("Model saved to admission_model.pkl")

# 11. Load Model Example
loaded_model = joblib.load('admission_model.pkl')
predict_admission(320, 110, 4, 4.5, 4.5, 9.0, 1)