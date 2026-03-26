from pathlib import Path

from src.preprocessing import load_kaggle_dataset, preprocess, resolve_dataset_path
from src.clustering import train_kmeans, save_model

# Load
dataset_path = resolve_dataset_path()
df = load_kaggle_dataset(dataset_path)

# Preprocess
df, features, scaler = preprocess(df)

# Train clustering
model, clusters = train_kmeans(features)

df["cluster"] = clusters

# Save outputs
Path("data").mkdir(parents=True, exist_ok=True)
df.to_csv("data/processed.csv", index=False)
save_model(model)

print("Training complete!")
print(f"Dataset source: {dataset_path}")
