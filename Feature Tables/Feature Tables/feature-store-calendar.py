# Databricks notebook source
# MAGIC %md # Calendar Feature Table
# MAGIC This notebook creates a calendar table with information about dates and us holidays:
# MAGIC - Day of the week
# MAGIC - Week/weekend flag
# MAGIC - Holiday flag

# COMMAND ----------

import pandas as pd
import holidays

from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup

from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DateType, BooleanType, IntegerType
from datetime import datetime, timedelta
import holidays

# COMMAND ----------

# MAGIC %md ## Create dataset

# COMMAND ----------



# Create a Spark session
spark = SparkSession.builder.appName("FeatureStore").getOrCreate()

# Define the schema for the feature table
feature_table_schema = StructType([
    StructField("date", DateType(), nullable=False),
    StructField("is_weekday", BooleanType(), nullable=False),
    StructField("is_weekend", BooleanType(), nullable=False),
    StructField("is_holiday", BooleanType(), nullable=False),
    StructField("day_of_week", IntegerType(), nullable=False),
])

# Generate data for the last year
start_date = datetime.now() - timedelta(days=365*5)
end_date = datetime.now() +timedelta(days=365*5)

date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

# Define US holidays
us_holidays = holidays.UnitedStates(years=[start_date.year, end_date.year])

data = []

for date in date_range:
    is_weekday = date.weekday() < 5
    is_weekend = not is_weekday
    is_holiday = date in us_holidays
    day_of_week = date.weekday()

    data.append((date, is_weekday, is_weekend, is_holiday, day_of_week))

# Create a DataFrame with the generated data and the defined schema
feature_table_data = spark.createDataFrame(data, feature_table_schema)

# Show the feature table
feature_table_data.show()

feature_table_data = feature_table_data.withColumn("date",feature_table_data["date"].cast('timestamp'))


# COMMAND ----------

# MAGIC %md ## Define the table name

# COMMAND ----------

# Create a unique table name for each run. This prevents errors if you run the notebook multiple times.
table_name = f"feature_store.dev.calendar"
print(table_name)

# COMMAND ----------

spark.sql('DROP TABLE if EXISTS feature_store.dev.calendar')

# COMMAND ----------

# MAGIC %md ## Create the feature table

# COMMAND ----------

# MAGIC %md The first step is to create a FeatureStoreClient.

# COMMAND ----------

# fs = feature_store.FeatureStoreClient()
fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md Create the feature table. For a complete API reference, see ([AWS](https://docs.databricks.com/machine-learning/feature-store/python-api.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/feature-store/python-api)|[GCP](https://docs.gcp.databricks.com/machine-learning/feature-store/python-api.html)).

# COMMAND ----------

fe.create_table(
    name=table_name,
    primary_keys=["date"],
    df=feature_table_data,
    description="date features"
)

# fe.write_table(
#     name=table_name,
#     df=feature_table_data,
#     mode='merge'
# )


# COMMAND ----------

