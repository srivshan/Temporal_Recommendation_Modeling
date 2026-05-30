📘 README 1 Temporal Recommendation Modeling Overview

This project builds a large-scale, time-aware recommendation system using 31.8M historical transactions to predict near-term customer purchases. The system models user–item interaction dynamics under strict temporal constraints and optimizes ranking quality using learning-to-rank objectives.

The full pipeline includes feature engineering, negative sampling, ranking model training, offline evaluation, and real-time inference serving.

📊 Dataset

31.8M transaction records

1.3M unique customers

Transaction fields: customer_id, article_id, t_dat, price, sales_channel_id

Temporal Setup

Last 7 days reserved for labeling

Historical data used for feature generation

Strict customer-level validation split

No look-ahead bias

System Architecture

Raw Transactions (31.8M)
↓
Label Construction (Hard Negative Sampling)
↓
Feature Engineering (PySpark)
↓
5.6M Ranking Training Samples
↓
LightGBM LambdaRank Model
↓
Evaluation (NDCG, Recall@K)
↓
FastAPI Inference (5ms latency)

⚙️ Feature Engineering

Built scalable behavioral and interaction features using PySpark from the full 31.8M transaction history.

Final training dataset contained 8 engineered ranking features, including:

Customer recency (days since last purchase)

Customer transaction frequency

Customer purchase diversity

Customer average price affinity

Item popularity

User–item interaction frequency

Interaction price deviation signal

Behavioral recency interactions

All features were generated strictly from historical data prior to the label window to prevent temporal leakage.

🤖 Model

Algorithm: LightGBM LambdaRank (Learning-to-Rank)

Objective:

Optimize ranking quality directly (NDCG@K)

Training dataset:

~5.6M labeled ranking pairs

Hard negative sampling (100 candidates per user)

📈 Evaluation Metrics
Metric	Value
NDCG@10	0.707
Recall@10	0.762
MAP@10	0.733
MRR@10	0.791

Performance improved from:

NDCG@10: 0.29 → 0.70
through interaction feature engineering.

🚀 Inference & Serving

Model deployed using FastAPI.

Latency Optimization

Initial inference latency:

296 ms per request (full-table scan bottleneck)

Optimized using indexed user grouping:

~5 ms total latency

~2.7 ms model inference time

Demonstrates production-aware serving optimization.

🧪 Reproducibility

Run pipeline:

python build_labels.py

python build_features.py

python train_ranker.py

uvicorn src.inference_api:app --reload
