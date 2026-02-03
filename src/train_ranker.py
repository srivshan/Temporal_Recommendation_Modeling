import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import json
import numpy as np

df = pd.read_parquet("data/processed/training_features.parquet")

df = df.sample(frac=0.3, random_state=42)

TARGET = "label"
DROP_COLS = ["customer_id", "article_id", "label"]

X = df.drop(columns=DROP_COLS)
y = df[TARGET]

group_sizes = (
    df.groupby("customer_id")
    .size()
    .values
)

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

X_train = train_df.drop(columns=DROP_COLS)
y_train = train_df["label"]

X_val = val_df.drop(columns=DROP_COLS)
y_val = val_df["label"]

train_group = train_df.groupby("customer_id").size().values
val_group = val_df.groupby("customer_id").size().values

FEATURE_COLUMNS = X_train.columns.tolist()


ranker = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    boosting_type="gbdt",
    num_leaves=63,
    learning_rate=0.05,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

ranker.fit(
    X_train,
    y_train,
    group=train_group,
    eval_set=[(X_val, y_val)],
    eval_group=[val_group],
    eval_at=[10]
)



with open("outputs/feature_columns.json", "w") as f:
    json.dump(FEATURE_COLUMNS, f)



def recall_at_k(df, model, k):
    recalls = []

    for cust_id, group in df.groupby("customer_id"):
        X_cust = group.drop(columns=DROP_COLS)
        y_true = group["label"].values

        scores = model.predict(X_cust)
        top_k_idx = np.argsort(scores)[-k:]

        recall = y_true[top_k_idx].sum() / max(1, y_true.sum())
        recalls.append(recall)

    return np.mean(recalls)

import numpy as np
import matplotlib.pyplot as plt


k_values = [1, 5, 10, 20, 50]
recall_scores = []

for k in k_values:
    r = recall_at_k(val_df, ranker, k=k)
    recall_scores.append(r)
    print(f"Recall@{k}: {r:.4f}")


plt.figure(figsize=(8,5))
plt.plot(k_values, recall_scores, marker='o')
plt.title("Recall@K vs K")
plt.xlabel("K")
plt.ylabel("Recall@K")
plt.grid(True)
plt.show()


import joblib
import os

os.makedirs("outputs", exist_ok=True)

joblib.dump(ranker, "outputs/lgbm_ranker.pkl")
