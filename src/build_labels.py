from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max as spark_max, date_sub, lit
from pyspark.sql.functions import rand

spark = (
    SparkSession.builder
    .appName("HM-Label-Building")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.driver.memory", "8g")
    .getOrCreate()
)


DATA_PATH = "data/processed/transactions.parquet"

df = spark.read.parquet(DATA_PATH)

max_date = df.select(spark_max("t_dat")).collect()[0][0]
print("Dataset max date:", max_date)

label_start_date = date_sub(lit(max_date), 7)

label_df = df.filter(col("t_dat") > label_start_date)
history_df = df.filter(col("t_dat") <= label_start_date)

positives = (
    label_df
    .select("customer_id", "article_id")
    .distinct()
    .withColumn("label", lit(1))
)




popular_articles = (
    history_df
    .groupBy("article_id")
    .count()
    .orderBy(col("count").desc())
    .limit(5000)
    .select("article_id")
)

customers = positives.select("customer_id").distinct()

negatives = (
    customers
    .crossJoin(popular_articles)
    .join(
        positives,
        on=["customer_id", "article_id"],
        how="left_anti"
    )
    .withColumn("label", lit(0))
    .sample(fraction=0.01, seed=42)
)

training_pairs = positives.unionByName(negatives)

OUTPUT_PATH = "data/processed/training_pairs.parquet"

(
    training_pairs
    .repartition(200)
    .write
    .mode("overwrite")
    .parquet(OUTPUT_PATH)
)

spark.stop()
