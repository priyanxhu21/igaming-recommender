from pathlib import Path

from sklearn.cluster import KMeans
import joblib

def train_kmeans(data, n_clusters=5):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = model.fit_predict(data)
    return model, clusters

def save_model(model, path="models/kmeans.pkl"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def load_model(path="models/kmeans.pkl"):
    return joblib.load(path)
