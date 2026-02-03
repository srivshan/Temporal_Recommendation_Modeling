from pyspark.sql import SparkSession
from pyspark.sql.functions import col, min, max, countDistinct

spark = (
    SparkSession.builder
    .appName("HM-Ingestion")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.driver.memory", "8g")
    .getOrCreate()
)
DATA_PATH = "data/transactions_train.csv"

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(DATA_PATH)
    .select(
        col("t_dat"),
        col("customer_id"),
        col("article_id"),
        col("price"),
        col("sales_channel_id")
    )
)

total_rows = df.count()
print("Total rows:", total_rows)


unique_customers = df.select("customer_id").distinct().count()
print("Unique customers:", unique_customers)


df_dates = df.select(
    min("t_dat").alias("min_date"),
    max("t_dat").alias("max_date")
)

df_dates.show()


OUTPUT_PATH = "data/processed/transactions.parquet"

(
    df
    .repartition(200)
    .write
    .mode("overwrite")
    .parquet(OUTPUT_PATH)
)


spark.stop()
