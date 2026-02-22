import pandas as pd
import lightgbm as lgb
import joblib
import json
import os
import numpy as np

df = pd.read_parquet("data/processed/training_features.parquet")

TARGET = "label"
DROP_COLS = [
    "customer_id",
    "article_id",
    "label",
    "last_txn_date"   # add this
]


# Split by customer
unique_customers = df["customer_id"].unique()
np.random.seed(42)
val_customers = np.random.choice(
    unique_customers,
    size=int(0.2 * len(unique_customers)),
    replace=False
)

train_df = df[~df["customer_id"].isin(val_customers)].copy()
val_df = df[df["customer_id"].isin(val_customers)].copy()

# IMPORTANT: sort before grouping
train_df = train_df.sort_values("customer_id")
val_df = val_df.sort_values("customer_id")

X_train = train_df.drop(columns=DROP_COLS)
y_train = train_df[TARGET]
X_val = val_df.drop(columns=DROP_COLS)
y_val = val_df[TARGET]

train_group = train_df.groupby("customer_id").size().values
val_group = val_df.groupby("customer_id").size().values

ranker = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    boosting_type="gbdt",

    # Increase model capacity
    num_leaves=63,              # more complex trees
    max_depth=-1,               # allow full depth
    min_child_samples=20,       # allow finer splits

    # More trees, smaller learning rate
    learning_rate=0.03,
    n_estimators=800,

    # Regularization
    subsample=0.9,
    colsample_bytree=0.9,

    # Ranking-specific improvement
    reg_lambda=1.0,
    reg_alpha=0.1,

    random_state=42,
    n_jobs=-1
)

ranker.fit(
    X_train,
    y_train,
    group=train_group,
    eval_set=[(X_val, y_val)],
    eval_group=[val_group],
    eval_at=[10],
    callbacks=[
        lgb.early_stopping(100),
        lgb.log_evaluation(50)
    ]
)


os.makedirs("outputs", exist_ok=True)

joblib.dump(ranker, "outputs/lgbm_ranker.pkl")

with open("outputs/feature_columns.json", "w") as f:
    json.dump(X_train.columns.tolist(), f)

X_val.to_parquet("data/processed/X_val.parquet")
y_val.to_frame(name="label").to_parquet("data/processed/y_val.parquet")

pd.Series(val_group).to_csv("data/processed/group_val.csv", index=False)

print("Training complete.")
