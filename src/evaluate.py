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
    recalls, precisions, maps, mrrs, ndcgs, hits = [], [], [], [], [], []

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


        precision = y_true_k.sum() / k
        precisions.append(precision)

 
        hit = 1.0 if y_true_k.sum() > 0 else 0.0
        hits.append(hit)

        ranks = np.where(y_true_k == 1)[0]
        mrr = 1 / (ranks[0] + 1) if len(ranks) > 0 else 0
        mrrs.append(mrr)

        ap = average_precision_score(y_true, y_score)
        maps.append(ap)


        ndcgs.append(
            ndcg_score([y_true], [y_score], k=k)
        )

    return {
        "Recall": np.mean(recalls),
        "Precision": np.mean(precisions),
        "Hit Rate": np.mean(hits),
        "MRR": np.mean(mrrs),
        "MAP": np.mean(maps),
        "NDCG": np.mean(ndcgs)
    }



results_10 = evaluate_k(val_df, k=10)


results_100 = evaluate_k(val_df, k=100)

print("=" * 40)
print(f"{'Metric':<15} {'@10':>10} {'@100':>10}")
print("=" * 40)
for metric in results_10:
    print(f"{metric:<15} {results_10[metric]:>10.4f} {results_100[metric]:>10.4f}")
print("=" * 40)
