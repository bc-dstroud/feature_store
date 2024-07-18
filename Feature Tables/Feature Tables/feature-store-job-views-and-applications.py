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

sdf = spark.sql(f"""
                with running_views as (
                select j.job_id, 
                    received_at, 
                    count(vjd.job_id) over(partition by j.job_id order by received_at) as running_views 
                from bc_foreign_snowflake.PRD_CMOS.VIEW_JOB_DETAILS vjd
                left join bc_foreign_snowflake.dm.dm_jobs j
                on vjd.job_id = j.external_job_id
                where received_at >= '2022-01-01'
                order by j.job_id, received_at
            ),
            running_applications as (
                select job_id, 
                    created_at as received_at, 
                    count(job_id) over(partition by job_id order by created_at) as running_applications
                from bc_foreign_snowflake.mysql_bluecrew._event_job_applied
                where created_at >= '2022-01-01'
                order by job_id, created_at
            ),
            views_and_apps as (
                select coalesce(a.job_id, v.job_id) as job_id,
                coalesce(a.received_at, v.received_at) as START_DATE,
                running_applications,
                running_views
                from running_applications a
                full outer join running_views v
                on a.job_id = v.job_id 
                and a.received_at = v.received_at
                where coalesce(a.job_id, v.job_id) is not null
                order by 1,2
            ),
            max_views_and_apps as (
            select job_id, START_DATE,
            max(running_applications) over(partition by job_id order by START_DATE) as max_apps,
            max(running_views) over(partition by job_id order by START_DATE) as max_views
            from views_and_apps
            )
            select job_id, 
                dateadd(HOUR,1,date_trunc('hour',START_DATE)) as START_DATE, 
                ifnull(max(max_views),0) as views,
                ifnull(max(max_apps),0) as apps 
            from max_views_and_apps 
            where job_id is not null
            group by 1,2
            order by 1,2
        
""")
display(sdf)

# COMMAND ----------

# Primary keys have to be integer type (not decimal)
sdf = sdf.withColumn("JOB_ID", sdf["JOB_ID"].cast('string')).withColumn("START_DATE",sdf["START_DATE"].cast('timestamp'))


# COMMAND ----------

# MAGIC %md ## Name Feature table

# COMMAND ----------

# Create a unique table name for each run. This prevents errors if you run the notebook multiple times.
table_name = f"feature_store.dev.job_views_and_applications"
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
    description="This contains the job view and application history."
)


# fe.write_table(
#     name=table_name,
#     df=sdf,
#     mode='merge'
# )


# COMMAND ----------

