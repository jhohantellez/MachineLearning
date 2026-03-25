import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("job_salary_prediction_dataset.csv", sep=";")

df.columns = df.columns.str.strip()

# variables
X = df[['experience_years', 'skills_count', 'certifications']]
y = df['salary']

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Modelo reentrenado")