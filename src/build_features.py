from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, datediff, max as spark_max, count, date_sub
)

spark = (
    SparkSession.builder
    .appName("HM-Feature-Engineering")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.driver.memory", "8g")
    .getOrCreate()
)

transactions = spark.read.parquet("data/processed/transactions.parquet")
pairs = spark.read.parquet("data/processed/training_pairs.parquet")

cutoff_date = transactions.select(spark_max("t_dat")).collect()[0][0]
history_df = transactions.filter(
    col("t_dat") <= date_sub(lit(cutoff_date), 7)
)

cust_last_txn = (
    history_df
    .groupBy("customer_id")
    .agg(spark_max("t_dat").alias("last_txn_date"))
    .withColumn(
        "cust_recency_days",
        datediff(lit(cutoff_date), col("last_txn_date"))
    )
    .select("customer_id", "cust_recency_days")
)

txn_30d = history_df.filter(
    col("t_dat") > date_sub(lit(cutoff_date), 30)
)

txn_90d = history_df.filter(
    col("t_dat") > date_sub(lit(cutoff_date), 90)
)

cust_freq_30d = txn_30d.groupBy("customer_id").agg(
    count("*").alias("cust_txn_count_30d")
)

cust_freq_90d = txn_90d.groupBy("customer_id").agg(
    count("*").alias("cust_txn_count_90d")
)

item_freq_30d = (
    txn_30d
    .groupBy("customer_id", "article_id")
    .agg(count("*").alias("item_freq_30d"))
)

item_freq_90d = (
    txn_90d
    .groupBy("customer_id", "article_id")
    .agg(count("*").alias("item_freq_90d"))
)

item_last_seen = (
    history_df
    .groupBy("customer_id", "article_id")
    .agg(spark_max("t_dat").alias("item_last_date"))
    .withColumn(
        "item_last_seen_days",
        datediff(lit(cutoff_date), col("item_last_date"))
    )
    .select("customer_id", "article_id", "item_last_seen_days")
)

item_pop_30d = txn_30d.groupBy("article_id").agg(
    count("*").alias("item_popularity_30d")
)

item_pop_90d = txn_90d.groupBy("article_id").agg(
    count("*").alias("item_popularity_90d")
)

features = (
    pairs
    .join(cust_last_txn, "customer_id", "left")
    .join(cust_freq_30d, "customer_id", "left")
    .join(cust_freq_90d, "customer_id", "left")
    .join(item_freq_30d, ["customer_id", "article_id"], "left")
    .join(item_freq_90d, ["customer_id", "article_id"], "left")
    .join(item_last_seen, ["customer_id", "article_id"], "left")
    .join(item_pop_30d, "article_id", "left")
    .join(item_pop_90d, "article_id", "left")
)

features.write.mode("overwrite").parquet(
    "data/processed/training_features.parquet"
)

spark.stop()
