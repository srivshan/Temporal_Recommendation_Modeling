from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    lit,
    datediff,
    max as spark_max,
    count,
    countDistinct,
    avg,
    broadcast,
    abs as spark_abs
)
from datetime import timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

transactions_path = BASE_DIR / "data/processed/transactions.parquet"
pairs_path = BASE_DIR / "data/processed/training_pairs.parquet"
output_path = BASE_DIR / "data/processed/training_features.parquet"

spark = (
    SparkSession.builder
    .appName("HM-Feature-Engineering-Advanced")
    .master("local[*]")
    .config("spark.driver.memory", "10g")
    .config("spark.sql.shuffle.partitions", "80")
    .getOrCreate()
)

print("Loading data...")

transactions = spark.read.parquet(str(transactions_path))
pairs = spark.read.parquet(str(pairs_path))

max_date = transactions.agg(spark_max("t_dat")).first()[0]
cutoff_date = max_date - timedelta(days=7)

history_df = transactions.filter(col("t_dat") <= cutoff_date)
history_df = history_df.persist()


print("Building customer features...")

cust_features = (
    history_df
    .groupBy("customer_id")
    .agg(
        spark_max("t_dat").alias("last_txn_date"),
        count("*").alias("cust_total_txn"),
        countDistinct("article_id").alias("cust_diversity"),
        avg("price").alias("cust_avg_price")
    )
    .withColumn(
        "cust_recency_days",
        datediff(lit(max_date), col("last_txn_date"))
    )
)


print("Building item features...")

item_features = (
    history_df
    .groupBy("article_id")
    .agg(
        count("*").alias("item_popularity"),
        avg("price").alias("item_avg_price")
    )
)

item_features = broadcast(item_features)


print("Building interaction features...")

user_item_freq = (
    history_df
    .groupBy("customer_id", "article_id")
    .count()
    .withColumnRenamed("count", "user_item_freq")
)


print("Joining features...")

features = (
    pairs
    .join(cust_features, "customer_id", "left")
    .join(item_features, "article_id", "left")
    .join(user_item_freq, ["customer_id", "article_id"], "left")
    .fillna(0)
)


features = features.withColumn(
    "price_diff_from_user_avg",
    spark_abs(col("item_avg_price") - col("cust_avg_price"))
)


features = features.repartition(40)

features.write \
    .mode("overwrite") \
    .option("compression", "snappy") \
    .parquet(str(output_path))

print("Advanced feature engineering complete.")

spark.stop()