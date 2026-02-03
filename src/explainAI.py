import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import os
import json


MODEL_PATH = "outputs/lgbm_ranker.pkl"
FEATURE_PATH = "data/processed/training_features.parquet"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("Loading model...")
ranker = joblib.load(MODEL_PATH)

print("Loading training features...")
X_train = pd.read_parquet(FEATURE_PATH)

X_train = X_train.drop(columns=["customer_id"])


with open("outputs/feature_columns.json") as f:
    FEATURE_COLUMNS = json.load(f)

X_train = X_train[FEATURE_COLUMNS]

shap_sample = X_train.sample(n=5000, random_state=42)

print("Running TreeSHAP...")
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

plt.savefig(
    f"{OUTPUT_DIR}/shap_global_importance.png",
    bbox_inches="tight"
)

print("SHAP analysis complete. Plot saved.")
