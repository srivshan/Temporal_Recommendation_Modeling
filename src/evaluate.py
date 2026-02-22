import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import average_precision_score, ndcg_score

model = joblib.load("outputs/lgbm_ranker.pkl")

X_val = pd.read_parquet("data/processed/X_val.parquet")
y_val = pd.read_parquet("data/processed/y_val.parquet")
group_val = pd.read_csv("data/processed/group_val.csv").values.flatten()


scores = model.predict(X_val)

customer_ids = np.repeat(np.arange(len(group_val)), group_val)

val_df = pd.DataFrame({
    "customer_id": customer_ids,
    "label": y_val["label"].values,
    "score": scores
})


def evaluate_k(df, k=10):
    recalls, maps, mrrs, ndcgs = [], [], [], []

    for cust_id, group in df.groupby("customer_id"):

        group = group.sort_values("score", ascending=False)

        y_true = group["label"].values
        y_score = group["score"].values

        if y_true.sum() == 0:
            continue

        y_true_k = y_true[:k]
        y_score_k = y_score[:k]

        recall = y_true_k.sum() / y_true.sum()
        recalls.append(recall)

        ranks = np.where(y_true_k == 1)[0]
        mrr = 1 / (ranks[0] + 1) if len(ranks) > 0 else 0
        mrrs.append(mrr)

        maps.append(
            average_precision_score(y_true_k, y_score_k)
        )

        ndcgs.append(
            ndcg_score([y_true], [y_score], k=k)
        )

    return (
        np.mean(recalls),
        np.mean(maps),
        np.mean(mrrs),
        np.mean(ndcgs)
    )

recall, map_k, mrr, ndcg = evaluate_k(val_df, k=10)

print(f"Recall@10: {recall:.4f}")
print(f"MAP@10: {map_k:.4f}")
print(f"MRR@10: {mrr:.4f}")
print(f"NDCG@10: {ndcg:.4f}")
