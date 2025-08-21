# Databricks notebook source
# 1) Read predictions from Delta
preds = spark.read\
             .format("delta")\
             .load("/delta/rossmann/predictions")

# 2) Read actuals
actuals = spark.table("rossmann_db.train")\
                .select("Date","Store","Sales")

# 3) Join & preview
df = (preds
       .join(actuals, on=["Date","Store"], how="inner")
       .select("Date","Store","Sales","Forecast"))

display(df.limit(1000))


# COMMAND ----------

from pyspark.sql.functions import expr

err_df = (df
          .withColumn("Error", expr("Forecast - Sales"))
          .groupBy("Date")
          .agg({"Error":"avg"})
          .withColumnRenamed("avg(Error)", "AvgError"))

display(err_df)


# COMMAND ----------

from pyspark.sql.functions import to_date, sum as _sum

# 1) Turn your timestamp into a pure date
daily = (
  df
    .withColumn("Day", to_date("Date"))
    .groupBy("Day")
    .agg(
      _sum("Sales").alias("Sales"),
      _sum("Forecast").alias("Forecast")
    )
    .orderBy("Day")
)

display(daily)


# COMMAND ----------

from pyspark.sql.functions import expr

err_df = (
  df
    .withColumn("Error", expr("Forecast - Sales"))
    .withColumn("Day", to_date("Date"))
    .groupBy("Day")
    .agg({"Error":"avg"})
    .withColumnRenamed("avg(Error)", "AvgError")
    .orderBy("Day")
)

display(err_df)
