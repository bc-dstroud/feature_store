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
job = 343455
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
    with all_times as (
    select distinct job_id, applied_status_start_date as start_date
    from bc_foreign_snowflake.dm.dm_cm_job_applied_history jah
    union all
    select j.job_id, j.job_created_at as start_date
    from bc_foreign_snowflake.dm.dm_jobs j
    union all
    select cr.job_id, cr.request_created_at as start_date
    from bc_foreign_snowflake.dm.dm_cm_shift_confirmation_requests cr
    where cr.request_created_at is not null
    union all
    select cr2.job_id, cr2.request_confirmed_at as start_date
    from bc_foreign_snowflake.dm.dm_cm_shift_confirmation_requests cr2
    where cr2.request_confirmed_at is not null
),
times_to_check as (
    select distinct at.job_id, at.start_date, j.job_start_date_time
    from all_times at
    inner join bc_foreign_snowflake.dm.dm_jobs j
    on at.job_id = j.job_id 
    where j.job_created_at >= '2022-01-01'
    order by 1, 2
)
    select ttc.job_id, ttc.start_date, count(jah2.job_id) as total_applicants, sum(case when cr.request_created_at is not null or datediff(ttc.job_start_date_time, jah2.applied_status_start_date) < 2 then 1 else 0 end) as requested, sum(case when cr.request_confirmed_at is not null or datediff(ttc.job_start_date_time, jah2.applied_status_start_date) < 2 then 1 else 0 end) as confirmed
    from times_to_check ttc
    left join bc_foreign_snowflake.dm.dm_cm_job_applied_history jah2
    on (ttc.job_id = jah2.job_id)
    and ttc.start_date >= jah2.applied_status_start_date 
    and ttc.start_date <jah2.end_date
    and jah2.applied_status_enum = 0
    left join bc_foreign_snowflake.dm.dm_cm_shift_confirmation_requests cr
    on jah2.user_id = cr.user_id 
    and jah2.job_id = cr.job_id 
    and ttc.start_date >= cr.request_created_at
    where (jah2.applied_status_enum = 0 or jah2.applied_status_start_date is null)
    group by ttc.job_id, ttc.start_date
    order by job_id, start_date

""") 
    # )
    # select job_id, start_date, count(*) as checker
    # from final
    # group by 1,2
    # order by 3 desc;
display(sdf)

# COMMAND ----------

# sdf = spark.sql(f"""
#                 with applicant_tracker as (
#         select 
#             case when applied_status_enum = 0 then 1 else -1 end as applicant_tracking, 
#             jah.*
#         from bc_foreign_snowflake.dm.dm_cm_job_applied_history jah
#         left join bc_foreign_snowflake.dm.dm_jobs j
#         on jah.job_id = j.job_id
#         where (jah.applied_status_enum in (0,7,8, 14) or (jah.applied_status_enum = 1 and j.job_is_application==1))
#         and j.job_created_at >= '{start_date}'
#         order by jah.user_id
#     ),
    
#     tracking as (    
#     select 
#     sum(applicant_tracking) over(partition by job_id order by at.applied_status_start_date) as total_applicants,
#     max(row_number() over(partition by job_id, at.applied_status_start_date order by applied_status_start_date)) over(partition by job_id, applied_status_start_date) as max_row,
#     row_number() over(partition by job_id, at.applied_status_start_date order by applied_status_start_date) as row_num,
#     at.*
#     from applicant_tracker at
#     order by at.applied_status_start_date
#     )
#     select coalesce(total_applicants,0) as total_applicants, coalesce(t.job_id,j.job_id) as job_id, coalesce(applied_status_start_date, j.job_created_at) as START_DATE
#     from tracking t
#     full outer join bc_foreign_snowflake.dm.dm_jobs j
#     on j.job_id = t.job_id and applied_status_start_date = j.job_created_at
#     where (row_num= max_row or (applied_status_start_date is null and j.job_created_at >= '{start_date}'))
#     order by job_id, start_date

# """) 
#     # )
#     # select job_id, start_date, count(*) as checker
#     # from final
#     # group by 1,2
#     # order by 3 desc;
# display(sdf)

# COMMAND ----------

# Primary keys have to be integer type (not decimal)
sdf = sdf.withColumn("JOB_ID", sdf["JOB_ID"].cast('string')).withColumn("START_DATE",sdf["START_DATE"].cast('timestamp'))


# COMMAND ----------

# MAGIC %md ## Name Feature table

# COMMAND ----------

# Create a unique table name for each run. This prevents errors if you run the notebook multiple times.
table_name = f"feature_store.dev.job_applicant_tracker"
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
    description="This contains the job applicant history."
)


# fe.write_table(
#     name=table_name,
#     df=sdf,
#     mode='merge'
# )


# COMMAND ----------

