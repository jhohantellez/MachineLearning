import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def getDataSet():
    df = pd.read_csv("BankChurners.csv")

    df = df[[
        "Customer_Age",
        "Credit_Limit",
        "Total_Trans_Amt",
        "Total_Trans_Ct"
    ]]

    df = df.dropna()

    return df


def AppClusteringKmeans(k=3):
    df = getDataSet()

    X = df.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    print(set(labels))

    df["cluster"] = labels

    results = df.to_dict(orient="records")

    summary = {}
    for label in labels:
        label = int(label)
        summary[label] = summary.get(label, 0) + 1

    centers = model.cluster_centers_.tolist()

    return {
        "results": results[:50],
        "summary": summary,
        "centers": centers
    }