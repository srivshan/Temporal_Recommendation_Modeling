from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max as spark_max, lit, rand
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
from datetime import timedelta

spark = (
    SparkSession.builder
    .appName("HM-Strong-Label-Building")
    .config("spark.driver.memory", "8g")
    .getOrCreate()
)

df = spark.read.parquet("data/processed/transactions.parquet")

# ------------------------------------------------
# Temporal Split
# ------------------------------------------------
max_date = df.agg(spark_max("t_dat")).first()[0]
label_start_date = max_date - timedelta(days=7)

label_df = df.filter(col("t_dat") > label_start_date)
history_df = df.filter(col("t_dat") <= label_start_date)

# ------------------------------------------------
# Positives
# ------------------------------------------------
positives = (
    label_df
    .select("customer_id", "article_id")
    .distinct()
    .withColumn("label", lit(1))
)

customers = positives.select("customer_id").distinct()

# ------------------------------------------------
# Popular Pool (Hard Negatives)
# ------------------------------------------------
popular_pool = (
    history_df
    .groupBy("article_id")
    .count()
    .orderBy(col("count").desc())
    .limit(3000)                     # bigger pool
    .select("article_id")
)

# ------------------------------------------------
# Random Pool (Tail Diversity)
# ------------------------------------------------
random_pool = (
    history_df
    .select("article_id")
    .distinct()
    .orderBy(rand())
    .limit(5000)
)

# ------------------------------------------------
# Function to Sample Negatives
# ------------------------------------------------
def sample_negatives(pool_df, n_per_user):

    candidate = (
        customers
        .join(pool_df)
        .join(positives, ["customer_id", "article_id"], "left_anti")
    )

    window = Window.partitionBy("customer_id").orderBy(rand())

    sampled = (
        candidate
        .withColumn("rn", row_number().over(window))
        .filter(col("rn") <= n_per_user)
        .drop("rn")
        .withColumn("label", lit(0))
    )

    return sampled


# 50 popular negatives
neg_popular = sample_negatives(popular_pool, 50)

# 50 random negatives
neg_random = sample_negatives(random_pool, 50)

# ------------------------------------------------
# Final Training Pairs
# ------------------------------------------------
training_pairs = positives.unionByName(neg_popular).unionByName(neg_random)

training_pairs.write.mode("overwrite").parquet(
    "data/processed/training_pairs.parquet"
)

spark.stop()
