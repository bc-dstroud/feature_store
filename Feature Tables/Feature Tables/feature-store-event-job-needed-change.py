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
start_date = '2022-01-01'
# end_date = '2023-11-01'

startdate = pd.to_datetime(start_date).date()
# enddate = pd.to_datetime(start_date).date()

# COMMAND ----------

# query = f"""

# select * from BLUECREW.PERSONALIZATION.USER_SNC_NCNS_CALENDAR
# where calendar_date >= '{start_date}'

# """
# # where job_start_date_time >= '{start_date}'
# sdf = spark.read.format("snowflake").options(**options).option("query", query).load()

# COMMAND ----------

sdf = spark.sql(f"""
                WITH tmp_needed AS ( -- Needed change table
                SELECT -- Get the current needed for all jobs
                    JOB_ID,
                    NEEDED AS OLD_NEEDED,
                    NEEDED AS NEW_NEEDED,
                    TO_TIMESTAMP_NTZ('2099-01-01') AS END_DATE
                FROM bc_foreign_mysql.bluecrew.JOBS_DATA
                UNION ALL
                SELECT -- Get the historic needed for all jobs with edits
                    JOB_ID,
                    OLD_NEEDED,
                    NEW_NEEDED,
                    TO_TIMESTAMP_NTZ(CREATED_AT) AS END_DATE
                FROM bc_foreign_mysql.bluecrew._EVENT_NUMBER_NEEDED_CHANGE
                WHERE OLD_NEEDED <> NEW_NEEDED
                ORDER BY JOB_ID, END_DATE),

                max_change as (
                    select job_id, END_DATE, old_needed, new_needed,
                    max(row_number() over(partition by job_id, end_date order by end_date)) over(partition by job_id, end_date) as max_id,
                    row_number() over(partition by job_id, end_date order by end_date) as id
                    from tmp_needed
                    where job_id in (select id from bc_foreign_mysql.bluecrew.jobs where created_at >= '{start_date}')
                ),
                tmp_needed2 as (
                select job_id, old_needed, new_needed, end_date
                from max_change
                where id = max_id
                )
                SELECT -- final table with job id and needed count by start/end date
                JOB_ID,
                OLD_NEEDED AS NEEDED,
                -- first row in lag get a far back start date
                LAG(END_DATE, 1, TO_TIMESTAMP('2001-01-01')) OVER (PARTITION BY JOB_ID ORDER BY END_DATE) AS START_DATE
            FROM tmp_needed2
            ORDER BY JOB_ID
""")
display(sdf)

# COMMAND ----------

# Primary keys have to be integer type (not decimal)
sdf = sdf.withColumn("JOB_ID", sdf["JOB_ID"].cast('string')).withColumn("START_DATE",sdf["START_DATE"].cast('timestamp'))


# COMMAND ----------

# MAGIC %md ## Name Feature table

# COMMAND ----------

# Create a unique table name for each run. This prevents errors if you run the notebook multiple times.
table_name = f"feature_store.dev.job_needed_change"
print(table_name)

# COMMAND ----------

# MAGIC %md ## Create the feature table

# COMMAND ----------

#why not featurestoreclient?
# fs = feature_store.FeatureStoreClient()

fe = FeatureEngineeringClient()

#


# COMMAND ----------

# MAGIC %md Create the feature table. For a complete API reference, see ([AWS](https://docs.databricks.com/machine-learning/feature-store/python-api.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/feature-store/python-api)|[GCP](https://docs.gcp.databricks.com/machine-learning/feature-store/python-api.html)).

# COMMAND ----------

spark.sql(f"""DROP TABLE if EXISTS {table_name}""")

# COMMAND ----------

fe.create_table(
    name=table_name,
    primary_keys=["JOB_ID", "START_DATE"],
    timestamp_keys="START_DATE",
    df=sdf,
    schema=sdf.schema,
    description="This contains the job needed history."
)


# fe.write_table(
#     name=table_name,
#     df=sdf,
#     mode='merge'
# )


# COMMAND ----------

