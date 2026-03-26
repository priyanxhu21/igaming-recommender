import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def recommend_games(user_input, df, feature_matrix, top_n=5):
    # Convert user input to vector
    user_vector = np.array(user_input).reshape(1, -1)

    # Compute similarity
    similarities = cosine_similarity(user_vector, feature_matrix)

    # Get top indices
    top_indices = similarities[0].argsort()[-top_n:][::-1]

    result_columns = ["game", "provider", "rtp", "volatility", "max_multiplier"]
    if "casino" in df.columns:
        result_columns.insert(1, "casino")

    results = df.iloc[top_indices][result_columns].copy()
    results["reason"] = "Similar RTP and feature profile match your preference"

    return results
