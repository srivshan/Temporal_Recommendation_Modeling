from pyspark.sql import SparkSession
from pyspark.sql.functions import col, min, max, to_date
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

spark = (
    SparkSession.builder
    .appName("HM-Ingestion")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.driver.memory", "8g")
    .getOrCreate()
)

schema = StructType([
    StructField("t_dat", StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("article_id", StringType(), True),
    StructField("price", DoubleType(), True),
    StructField("sales_channel_id", IntegerType(), True),
])

df = (
    spark.read
    .option("header", True)
    .schema(schema)
    .csv("data/transactions_train.csv")
)

df = df.withColumn("t_dat", to_date(col("t_dat")))

df.cache()

print("Total rows:", df.count())
print("Unique customers:", df.select("customer_id").distinct().count())

df.select(min("t_dat"), max("t_dat")).show()

df.write.mode("overwrite").parquet("data/processed/transactions.parquet")

spark.stop()
