import pandas as pd
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc)

def get_trained_model():
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
    df.columns = ['label', 'message']
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['message'])
    y = df['label_num']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    os.makedirs('static/images', exist_ok=True)

    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.savefig('static/images/confusion_matrix_nb.png')
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    metrics = {
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred), 4),
        'recall': round(recall_score(y_test, y_pred), 4),
        'f1': round(f1_score(y_test, y_pred), 4),
        'auc': round(auc(fpr, tpr), 4)
    }
    
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"AUC = {metrics['auc']}")
    plt.plot([0, 1], [0, 1], '--')
    plt.savefig('static/images/roc_curve_nb.png')
    plt.close()

    return model, vectorizer, metrics