# Databricks notebook source
# 1) Imports
from pyspark.sql.functions import (
    dayofweek, weekofyear, month, year,
    date_format, lag, col
)
from pyspark.sql import Window
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler


# COMMAND ----------

# 2) Load your train and store tables
train_df = spark.read.format("delta").load("/delta/rossmann/raw/train")
store_df = spark.read.format("delta").load("/delta/rossmann/raw/store")

# COMMAND ----------

# 3) Basic temporal features - sesonality, trends, events
feat_df = (
    train_df
    .withColumn("dow", dayofweek("Date"))                   # 1=Sunday…7=Saturday
    .withColumn("woy", weekofyear("Date"))                  # week of year
    .withColumn("moy", month("Date"))                       # month of year
    .withColumn("y",   year("Date"))                        # numeric year
)

# COMMAND ----------

display(feat_df)

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql.functions import lag, expr

# 4) Lag features: previous day’s sales, 7-day avg
w = Window.partitionBy("Store").orderBy(col("Date"))
feat_df = (
    feat_df
    .withColumn("lag_1",   lag("Sales", 1).over(w))                             # yesterday
    .withColumn("lag_7",   lag("Sales", 7).over(w))                             # one-week lag
    .withColumn("rolling_7_avg", 
                expr("avg(Sales) OVER (PARTITION BY Store ORDER BY Date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)"))
)

# COMMAND ----------

feat_df = feat_df.join(store_df, on="Store", how="left")
display(feat_df)

# COMMAND ----------

# Categorical encoding
indexers = [
    StringIndexer(inputCol="StoreType", outputCol="StoreType_idx"),
    StringIndexer(inputCol="Assortment", outputCol="Assortment_idx"),
]
encoders = [
    OneHotEncoder(inputCol="StoreType_idx",   outputCol="StoreType_vec"),
    OneHotEncoder(inputCol="Assortment_idx",  outputCol="Assortment_vec"),
]
for idx in indexers: feat_df = idx.fit(feat_df).transform(feat_df)
for enc in encoders: feat_df = enc.fit(feat_df).transform(feat_df)

# COMMAND ----------

final_df.printSchema()

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# List out exactly the columns you engineered
inputCols = [
  "dow", "woy", "moy",
  "lag_1", "lag_7", "rolling_7_avg",
  "Promo", "CompetitionDistance",
  "StoreType_vec", "Assortment_vec"
]

# Create an assembler that skips any row where one of these inputs is null
assembler = VectorAssembler(
    inputCols=inputCols,
    outputCol="features",
    handleInvalid="skip"
)

# Transform your DataFrame to add the features column
final_df = assembler.transform(feat_df)

# Quick sanity check—show you have a non-null vector and the Sales label
display(final_df.select("Date","Store","features","Sales").limit(5))


# COMMAND ----------

# Write out only the essential columns for modeling
(
  final_df
    .select("Date","Store","features","Sales")
    .write
    .mode("overwrite")
    .format("delta")
    .save("/delta/rossmann/features")
)

# Register it in the metastore
spark.sql("""
  CREATE TABLE IF NOT EXISTS rossmann_db.features
  USING DELTA
  LOCATION '/delta/rossmann/features'
""")
