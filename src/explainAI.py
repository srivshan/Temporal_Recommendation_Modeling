import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import os
import json
import numpy as np

MODEL_PATH = "outputs/lgbm_ranker.pkl"
FEATURE_PATH = "data/processed/training_features.parquet"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading model...")
ranker = joblib.load(MODEL_PATH)

print("Loading features...")
df = pd.read_parquet(FEATURE_PATH)

with open("outputs/feature_columns.json") as f:
    FEATURE_COLUMNS = json.load(f)

# Keep full dataframe for ranking analysis
X = df[FEATURE_COLUMNS]
customer_ids = df["customer_id"]

# ------------------------------------------------
# 1️⃣ GLOBAL SHAP IMPORTANCE
# ------------------------------------------------

print("Running global SHAP analysis...")

shap_sample = X.sample(n=5000, random_state=42)
explainer = shap.TreeExplainer(ranker)
shap_values = explainer.shap_values(shap_sample)

plt.figure()
shap.summary_plot(
    shap_values,
    shap_sample,
    plot_type="bar",
    max_display=15,
    show=False
)

plt.savefig(f"{OUTPUT_DIR}/shap_global_importance.png", bbox_inches="tight")
plt.close()

print("Global SHAP saved.")


# ------------------------------------------------
# 2️⃣ PER-CUSTOMER TOP-K EXPLANATION
# ------------------------------------------------

print("Generating per-customer ranking explanation...")

df["score"] = ranker.predict(X)

# Pick a random customer
sample_customer = df["customer_id"].sample(1, random_state=42).values[0]
customer_df = df[df["customer_id"] == sample_customer].copy()

customer_df = customer_df.sort_values("score", ascending=False)

top_k = customer_df.head(5)

print(f"\nTop 5 recommendations for customer {sample_customer}")
print(top_k[["article_id", "score"]])

# SHAP for top 5
top_k_X = top_k[FEATURE_COLUMNS]
top_k_shap = explainer.shap_values(top_k_X)

for i in range(len(top_k)):
    plt.figure()
    shap.force_plot(
        explainer.expected_value,
        top_k_shap[i],
        top_k_X.iloc[i],
        matplotlib=True,
        show=False
    )
    plt.savefig(f"{OUTPUT_DIR}/customer_{sample_customer}_rank_{i+1}.png",
                bbox_inches="tight")
    plt.close()

print("Per-customer explanations saved.")


# ------------------------------------------------
# 3️⃣ RANK DIFFERENCE ANALYSIS (Why Rank1 > Rank2)
# ------------------------------------------------

if len(top_k) >= 2:
    print("\nAnalyzing rank difference between top 2 items...")

    diff = top_k_shap[0] - top_k_shap[1]
    diff_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "shap_difference": diff
    }).sort_values("shap_difference", ascending=False)

    diff_df.to_csv(f"{OUTPUT_DIR}/rank_comparison.csv", index=False)

    print("Rank comparison saved.")


# ------------------------------------------------
# 4️⃣ Cold-Start Segment Analysis
# ------------------------------------------------

if "cust_total_txn" in df.columns:
    cold_start = df[df["cust_total_txn"] < 3]
    print(f"\nCold-start rows: {len(cold_start)}")

print("\nExplainability pipeline complete.")
