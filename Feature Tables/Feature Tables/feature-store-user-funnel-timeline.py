# Databricks notebook source
# MAGIC %md # Feature Table for Job Attributes
# MAGIC
# MAGIC This notebook builds a job feature table with information about:
# MAGIC - Average Job Wage for the same region, job_type, and job_title in a given year and month
# MAGIC - Difference in Job Wage for the current job
# MAGIC - Number of eligible CMs in the region on the job start date
# MAGIC - Job posting lead time
# MAGIC

# COMMAND ----------

import pandas as pd

from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup
from databricks.feature_engineering import FeatureEngineeringClient


from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DateType, BooleanType, IntegerType
from datetime import datetime, timedelta


# COMMAND ----------

# MAGIC %md ## Load dataset
# MAGIC The code in the following cell loads the dataset and does some minor data preparation: creates a unique ID for each observation and removes spaces from the column names. The unique ID column (`wine_id`) is the primary key of the feature table and is used to lookup features.

# COMMAND ----------

sfUser = dbutils.secrets.get(scope="my_secrets", key="snowflake-user")
SFPassword = dbutils.secrets.get(scope="my_secrets", key="snowflake-password")
 
options = {
  "sfUrl": "vha09841.snowflakecomputing.com",
  "sfUser": sfUser,
  "SFPassword": SFPassword,
  "sfDataBase": "BLUECREW",
  "sfSchema": "PERSONALIZATION",
  "sfWarehouse": "COMPUTE_WH"
}
start_date = '2020-01-01'
# end_date = '2023-11-01'

startdate = pd.to_datetime(start_date).date()
# enddate = pd.to_datetime(start_date).date()

# COMMAND ----------

query = f"""

select * from BLUECREW.DM.DM_CM_FUNNEL
where user_id is not null
"""

sdf = spark.read.format("snowflake").options(**options).option("query", query).load()

# COMMAND ----------

from pyspark.sql.functions import expr
import pyspark.sql.functions as F
from pyspark.sql.functions import greatest

sdf2 = sdf
# cols = []
sdf2 = sdf2.withColumn("max_date", greatest('USER_SIGNUP_DATE','PHONE_VERIFIED_DATE', 'BROWSED_FOR_JOBS_DATE', 'JOBS_AVAILABLE_DATE', 'SELECTED_FIRST_JOB_DATE', 'COMPLETE_WELCOME_VIDEO_DATE', 'COMPLETE_PE_VIDEO_DATE', 'PROVIDE_EMERGENCY_CONTACT_DATE', 'COMPLETE_SAFETY_TRAINING_DATE', 'COMPLETED_COVID_19_QUESTION_DATE', 'I9_SECTION_1_DATE', 'SELECT_I9_AGENT_DATE', 'I9_SECTION_2_DATE', 'COMPLETE_BACKGROUND_CHECK_DATE', 'PASS_BACKGROUND_CHECK_DATE', 'ELIGIBLE_DATE', 'FIRST_JOB_PLACEMENT_DATE', 'FIRST_SHIFT_WORKED_DATE'))
sdf2 = sdf2.na.drop(subset=["max_date"])
for col in sdf2.columns:
  if col not in ['USER_ID', 'USER_SIGNUP_DATE', 'I9_SECTION_1_SOURCE',"I9_SECTION_2_SOURCE", "ETL_DATE",'max_date']:
    # cols.append(col)
    sdf2 = sdf2.withColumn(col[:-4]+'DAYS',F.date_diff(sdf2[f"{col}"],sdf2.USER_SIGNUP_DATE))
    sdf2 = sdf2.withColumn(col[:-4]+'MINS', (F.col(f"{col}").cast("long") - F.col("USER_SIGNUP_DATE").cast("long"))/60.
)
    sdf2=sdf2.drop(col)
sdf2 = sdf2.drop('ETL_DATE')
sdf2 = sdf2.withColumn('ever_worked', F.when(F.col('FIRST_SHIFT_WORKED_MINS').isNotNull(), 1).otherwise(0))


from pyspark.sql.window import Window

windowSpec = Window.partitionBy("USER_ID").orderBy(F.desc("USER_SIGNUP_DATE"))
sdf2 = sdf2.withColumn("rank", F.row_number().over(windowSpec))
sdf2 = sdf2.filter(sdf2.rank == 1)
sdf2 = sdf2.drop("rank")

# print(cols)
display(sdf2)

# COMMAND ----------

# Primary keys have to be integer type (not decimal)
sdf2 = sdf2.withColumn("USER_ID", sdf2["USER_ID"].cast('string'))


# COMMAND ----------

# MAGIC %md ## Name Feature table

# COMMAND ----------

# Create a unique table name for each run. This prevents errors if you run the notebook multiple times.
table_name = f"feature_store.dev.user_funnel_timeline"
print(table_name)

# COMMAND ----------

# MAGIC %md ## Create the feature table

# COMMAND ----------

#why not featurestoreclient?
# fs = feature_store.FeatureStoreClient()

fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md Create the feature table. For a complete API reference, see ([AWS](https://docs.databricks.com/machine-learning/feature-store/python-api.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/feature-store/python-api)|[GCP](https://docs.gcp.databricks.com/machine-learning/feature-store/python-api.html)).

# COMMAND ----------

spark.sql(f"""DROP TABLE if exists {table_name}""")

# COMMAND ----------

fe.create_table(
    name=table_name,
    primary_keys=["USER_ID", "max_date"],
    timestamp_keys="max_date",
    df=sdf2["USER_ID", "max_date", 'ever_worked'],
    description="This contains some calculated features about each user's sign up timeline."
)


# fe.write_table(
#     name=table_name,
#     df=sdf,
#     mode='merge'
# )


# COMMAND ----------

# MAGIC %sql
# MAGIC select * from feature_store.dev.user_funnel_timeline
# MAGIC where USER_ID = 47502

# COMMAND ----------

