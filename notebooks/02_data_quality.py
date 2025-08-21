# Databricks notebook source
import great_expectations as ge
from great_expectations.dataset import SparkDFDataset

# COMMAND ----------

# 1) List all databases
spark.sql("SHOW DATABASES").show()

# 2) List tables in rossmann_db (if it exists)
spark.sql("SHOW TABLES IN rossmann_db").show()


# COMMAND ----------

spark_df = spark.table("rossmann_db.train")
display(spark_df.limit(5))
spark_df.printSchema()

# COMMAND ----------

ge_df = SparkDFDataset(spark_df)

# COMMAND ----------

# 4.1: No missing dates or store IDs
ge_df.expect_column_values_to_not_be_null("Date")
ge_df.expect_column_values_to_not_be_null("Store")

# 4.2: Sales and Customers are non-negative (and at least 1 customer if store is open)
ge_df.expect_column_values_to_be_between("Sales", min_value=0)
ge_df.expect_column_values_to_be_between("Customers", min_value=0)

# 4.3: DayOfWeek is in the 1–7 range
ge_df.expect_column_values_to_be_between("DayOfWeek", min_value=1, max_value=7)

# 4.4: Open and Promo flags are binary
ge_df.expect_column_values_to_be_in_set("Open", [0, 1])
ge_df.expect_column_values_to_be_in_set("Promo", [0, 1])

# 4.5: StateHoliday uses only allowed codes (0 = none, a/b/c = holiday types)
ge_df.expect_column_values_to_be_in_set("StateHoliday", ["0", "a", "b", "c"])

# 4.6: CompetitionDistance, if present, is non-negative
cols = spark_df.columns 
if "CompetitionDistance" in cols:
    ge_df.expect_column_values_to_be_between("CompetitionDistance", min_value=0)


# COMMAND ----------

# Execute all expectations and return a summary
results = ge_df.validate(result_format="SUMMARY")

# Display the JSON-esque summary of successes/failures
display(results)

# COMMAND ----------

