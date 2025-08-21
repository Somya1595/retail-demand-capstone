# Databricks notebook source
display(dbutils.fs.ls("/FileStore/tables"))

# COMMAND ----------

raw_train = "/FileStore/tables/train.csv"
raw_test = "/FileStore/tables/test.csv"
raw_store = "/FileStore/tables/store.csv"

# COMMAND ----------

from pyspark.sql.functions import to_date, year, month, col

base_delta = "/delta/rossmann/raw"

# 1) Ingest train.csv

train_df = (
    spark.read
    .option("header", True).option("inferSchema", True)
    .csv(raw_train)
    .withColumn("Date", to_date(col("Date"),"yyyy-MM-dd"))
    .withColumn("month", month("Date"))
    .withColumn("year", year("Date"))
            )

train_df.write\
    .mode("overwrite")\
    .partitionBy("year", "month")\
    .format("delta")\
    .save(f"{base_delta}/train")

# COMMAND ----------

# 2) Ingest store.csv
store_df = (
  spark.read
       .option("header", True).option("inferSchema", True)
       .csv(raw_store)
)
store_df.write \
        .mode("overwrite") \
        .format("delta") \
        .save(f"{base_delta}/store")

# 3) Ingest test.csv
test_df = (
  spark.read
       .option("header", True).option("inferSchema", True)
       .csv(raw_test)
       .withColumn("Date", to_date(col("Date"), "yyyy-MM-dd"))
       .withColumn("year", year("Date"))
       .withColumn("month", month("Date"))
)
test_df.write \
       .mode("overwrite") \
       .partitionBy("year","month") \
       .format("delta") \
       .save(f"{base_delta}/test")

# COMMAND ----------

# 4) Create metastore tables
spark.sql("CREATE DATABASE IF NOT EXISTS rossmann_db")
for tbl in ["train", "store", "test"]:
    spark.sql(f"""
              CREATE TABLE IF NOT EXISTS rossmann_db.{tbl}
              USING DELTA
              LOCATION '{base_delta}/{tbl}'
              """)