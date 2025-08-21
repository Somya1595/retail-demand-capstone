# Databricks notebook source
import mlflow.spark
from pyspark.sql.functions import col
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# 1) Load your best RF model
model_uri   = "runs:/92caff057d0146faa4a7fdb304e5efdc/best_rf_model"
loaded_model = mlflow.spark.load_model(model_uri)

# 2) Read in the feature table and the raw test set
features_df = spark.table("rossmann_db.features")
to_score    = features_df.select("Date","Store","features")


# COMMAND ----------



# 4) Run the model
preds = loaded_model.transform(to_score)


# COMMAND ----------

# 5) Write out only Date, Store, and your new Forecast column
(preds
  .select("Date","Store", col("prediction").alias("Forecast"))
  .write
  .mode("overwrite")
  .format("delta")
  .save("/delta/rossmann/predictions")
)


# COMMAND ----------

spark.sql("CREATE DATABASE IF NOT EXISTS rossmann_db")
spark.sql("""
  CREATE TABLE IF NOT EXISTS rossmann_db.predictions
  USING DELTA
  LOCATION '/delta/rossmann/predictions'
""")