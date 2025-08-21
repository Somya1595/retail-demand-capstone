# Databricks notebook source


import mlflow
import mlflow.spark
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


# COMMAND ----------

mlflow.spark.autolog()

# COMMAND ----------

mlflow.set_experiment("/Users/databricks@zirkeltech.com/rossmann_demand_forecast")

# COMMAND ----------

import mlflow
exp = mlflow.get_experiment_by_name("/Users/databricks@zirkeltech.com/rossmann_demand_forecast")
print(exp)


# COMMAND ----------

features_df = spark.table("rossmann_db.features")
train_df, test_df = features_df.randomSplit([0.8,0.2], seed=42)
print(f"Training rows: {train_df.count()}, Test rows: {test_df.count()}")

# COMMAND ----------

# Run a Baseline Random Forest
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(labelCol="Sales", predictionCol="prediction", metricName="rmse")

with mlflow.start_run(run_name="rf_baseline"):
    # 1) Define & train
    rf = RandomForestRegressor(featuresCol="features", labelCol="Sales", numTrees=20, maxDepth=5)
    model = rf.fit(train_df)

    # 2) Predict & evaluate
    preds = model.transform(test_df)
    rmse = evaluator.evaluate(preds)

    # 3) Log a custom metric (autolog will handle params & model)
    mlflow.log_metric("rmse_baseline", rmse)

    print(f"Baseline RF RMSE: {rmse:.2f}")


# COMMAND ----------

# Hyperparameter Tuning with CrossValidator

# 1) Build a param grid
paramGrid = (ParamGridBuilder()
             .addGrid(rf.numTrees, [20, 50])
             .addGrid(rf.maxDepth, [5, 10])
             .build())

# 2) Set up 3-fold CV
cv = CrossValidator(
    estimator=rf,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=2,
    parallelism=2  # adjust based on your cluster size
)

with mlflow.start_run(run_name="rf_cv"):
    # 3) Train with cross-validation
    cv_model = cv.fit(train_df)
    best_model = cv_model.bestModel

    # 4) Score & log final metric
    cv_preds = best_model.transform(test_df)
    rmse_cv = evaluator.evaluate(cv_preds)
    mlflow.log_metric("rmse_cv", rmse_cv)

    # 5) Log the tuned model
    mlflow.spark.log_model(best_model, "best_rf_model")

    print(f"Tuned RF RMSE: {rmse_cv:.2f}")


# COMMAND ----------


model_uri = "runs:/92caff057d0146faa4a7fdb304e5efdc/best_rf_model"
loaded_model = mlflow.spark.load_model(model_uri)


# COMMAND ----------


import mlflow.spark
model = mlflow.spark.load_model(model_uri)
preds = model.transform(
    spark.table("rossmann_db.features")
         .limit(5)
         .select("features")
)
display(preds)
