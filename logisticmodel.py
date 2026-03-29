import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report , auc , roc_curve

data = pd.read_csv('loan_prediction_dataset.csv')

print(data.head())

x = data.drop('Loan_Approved', axis=1)
y = data['Loan_Approved']

x = pd.get_dummies(x, columns=['Employment_Status'])

x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=42)

scaler= StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


logistic_model = LogisticRegression()
logistic_model.fit(x_train_scaled, y_train)

y_pred = logistic_model.predict(x_test_scaled)

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap= 'Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print( classification_report(y_test, y_pred))

accuracy= accuracy_score( y_test, y_pred)
print(f'Exactitud del modelo: {accuracy * 100: .2f}%')

def pre_resultado(datos_entrada):
    df = pd.DataFrame([datos_entrada])
    df = pd.get_dummies(df)
    df = df.reindex(columns=x.columns, fill_value=0)
    datos_escalados = scaler.transform(df)

    prediccion = logistic_model.predict(datos_escalados)[0]
    probabilidad = logistic_model.predict_proba(datos_escalados)[0][1]

    return int(prediccion), probabilidad

y_prob = logistic_model.predict_proba(x_test_scaled)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

if __name__ == "__main__":
    print("ready model")