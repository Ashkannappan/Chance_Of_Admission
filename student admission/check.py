import pandas as pd

data = pd.read_csv('Admission_Predict.csv')
print(data.columns)


import joblib
import pandas as pd

# Load the trained model
model = joblib.load('admission_model.pkl')

# Example: Predict admission chance for a new applicant
input_data = pd.DataFrame([[330, 115, 5, 5, 5, 9.5, 1]],
                          columns=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research'])

prediction = model.predict(input_data)[0]
print(f'Predicted Chance of Admission: {prediction:.2f}')
print(model)