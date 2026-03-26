import matplotlib.pyplot as plt
import pandas as pd
import joblib


pipeline = joblib.load("model.pkl")


df = pd.read_csv("job_salary_prediction_dataset.csv", sep=";")
df.columns = df.columns.str.strip()

variables = ['experience_years', 'skills_count', 'certifications']

for var in variables:
    X_var = df[[var]]
    
    y_pred = pipeline.predict(df[['experience_years', 'skills_count', 'certifications']])
    
    plt.figure()
    plt.scatter(df[var], df['salary'], color='blue', label='Datos reales')
    plt.plot(df[var], y_pred, color='red', label='Predicción')
    plt.xlabel(var)
    plt.ylabel('Salary')
    plt.title(f'Salary vs {var}')
    plt.legend()
    plt.show()