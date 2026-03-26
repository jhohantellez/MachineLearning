import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_csv("job_salary_prediction_dataset.csv", sep=";")
df.columns = df.columns.str.strip()

X = df[['experience_years', 'skills_count', 'certifications']]
y = df['salary']


pipeline = Pipeline([
    ('scaler', StandardScaler()),      
    ('regressor', LinearRegression())  
])


pipeline.fit(X, y)

joblib.dump(pipeline, "model.pkl")

print("Modelo reentrenado con normalización")