from fastapi import FastAPI
import joblib
import pandas as pd
import json
import time

app = FastAPI()

model = joblib.load("outputs/lgbm_ranker.pkl")

with open("outputs/feature_columns.json") as f:
    FEATURE_COLUMNS = json.load(f)

features = pd.read_parquet("data/processed/training_features.parquet")

# 🔥 Pre-group users for O(1) lookup (huge improvement)
user_groups = dict(tuple(features.groupby("customer_id")))


@app.get("/recommend/{customer_id}")
def recommend(customer_id: str, top_k: int = 10):

    total_start = time.perf_counter()

    # 1️⃣ Data lookup latency
    lookup_start = time.perf_counter()
    user_data = user_groups.get(customer_id)
    lookup_latency = (time.perf_counter() - lookup_start) * 1000

    if user_data is None:
        return {"error": "Customer not found"}

    X = user_data[FEATURE_COLUMNS]

    # 2️⃣ Model inference latency
    model_start = time.perf_counter()
    scores = model.predict(X)
    model_latency = (time.perf_counter() - model_start) * 1000

    user_data = user_data.copy()
    user_data["score"] = scores

    top_items = (
        user_data
        .sort_values("score", ascending=False)
        .head(top_k)
        [["article_id", "score"]]
    )

    total_latency = (time.perf_counter() - total_start) * 1000

    return {
        "customer_id": customer_id,
        "lookup_latency_ms": round(lookup_latency, 2),
        "model_latency_ms": round(model_latency, 2),
        "total_latency_ms": round(total_latency, 2),
        "recommendations": top_items.to_dict(orient="records")
    }