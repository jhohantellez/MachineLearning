import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def getDataSet():
    df = pd.read_csv("BankChurners.csv")

    df = df[[
        "Credit_Limit",
        "Total_Trans_Amt",
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

    df["cluster"] = labels

    summary = {}
    for label in labels:
        label = int(label)
        summary[label] = summary.get(label, 0) + 1

    centers = model.cluster_centers_.tolist()

    grouped = df.groupby("cluster")

    sampled = []

    for cluster_id, group in grouped:
        sample = group.sample(min(10, len(group)), random_state=42)
        sampled.extend(sample.to_dict(orient="records"))

    plt.figure()

    plt.scatter(
        df["Credit_Limit"],
        df["Total_Trans_Amt"],
        c=df["cluster"]
    )

    plt.xlabel("Credit Limit")
    plt.ylabel("Total Amount")

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)

    image_png = buffer.getvalue()
    buffer.close()

    graph = base64.b64encode(image_png).decode("utf-8")

    plt.close()

    return {
        "results": sampled,
        "summary": summary,
        "centers": centers,
        "graph": graph
    }