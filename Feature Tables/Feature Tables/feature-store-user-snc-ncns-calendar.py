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

sdf = spark.sql(f"""with user_dates as (
    select coalesce(ncns.user_id,snc.user_id) as user_id,
    min(NO_SHOW_SHIFT_START_TIME_UTC) as ncns_min_date,
    max(NO_SHOW_SHIFT_START_TIME_UTC) as ncns_max_date,
    min(CANCEL_TIMESTAMP_UTC) as snc_min_date,
    max(CANCEL_TIMESTAMP_UTC) as snc_max_date
    from bc_foreign_snowflake.dm.dm_cm_no_call_no_show ncns
    full outer join bc_foreign_snowflake.dm.dm_cm_short_notice_cancellations snc
    on ncns.user_id = snc.user_id
    -- where shift_start >= '2018-01-01'
    and (NO_SHOW_SHIFT_START_TIME_UTC >= '2018-01-01' or NO_SHOW_SHIFT_START_TIME_UTC is null)
    and (CANCEL_TIMESTAMP_UTC >= '2018-01-01' or CANCEL_TIMESTAMP_UTC is null)
    group by 1
), 
ncns_snc_calendar as (
SELECT calendar_date, 
    ud.user_id, 
    calendar_week,
    calendar_day_of_week,
    count(distinct ncns.segment_id) as ncns_shifts, 
    count(distinct snc.snc_index) as snc_shifts
FROM bc_foreign_snowflake.DM.DM_CALENDAR_DAYS 
INNER JOIN USER_DATES ud
ON (calendar_date >=snc_min_date or calendar_date >=ncns_min_date)
AND (calendar_date <= snc_max_date or calendar_date <=ncns_max_date)
LEFT JOIN bc_foreign_snowflake.DM.dm_cm_short_notice_cancellations snc
ON DATE_TRUNC('DAY', snc.CANCEL_TIMESTAMP_UTC) = CALENDAR_DATE
AND snc.user_id = ud.user_id
LEFT JOIN bc_foreign_snowflake.DM.dm_cm_no_call_no_show ncns
ON DATE_TRUNC('DAY', ncns.NO_SHOW_SHIFT_START_TIME_UTC) = CALENDAR_DATE
AND ncns.user_id = ud.user_id
group by 1, 2, 3, 4
order by 2, 1
),
SNC_Cal as (
select 
    calendar_date,
    user_id,
    ncns_shifts as NCNS_EVENT,
    snc_shifts as SNC_EVENT,
    SUM(NCNS_SHIFTS) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as NCNS_SHIFTS_TOTAL,
    SUM(NCNS_SHIFTS) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN 30 PRECEDING AND CURRENT ROW) as NCNS_SHIFTS_LAST_30_DAYS,
    SUM(NCNS_SHIFTS) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN 90 PRECEDING AND CURRENT ROW) as NCNS_SHIFTS_LAST_90_DAYS,
    SUM(NCNS_SHIFTS) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN 365 PRECEDING AND CURRENT ROW) as NCNS_SHIFTS_LAST_365_DAYS,
    SUM(SNC_SHIFTS) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as SNC_SHIFTS_TOTAL,
    SUM(SNC_SHIFTS) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN 30 PRECEDING AND CURRENT ROW) as SNC_SHIFTS_LAST_30_DAYS,
    SUM(SNC_SHIFTS) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN 90 PRECEDING AND CURRENT ROW) as SNC_SHIFTS_LAST_90_DAYS,
    SUM(SNC_SHIFTS) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN 365 PRECEDING AND CURRENT ROW) as SNC_SHIFTS_LAST_365_DAYS
from ncns_snc_calendar
order by user_id desc, calendar_date
)


select * from SNC_cal where calendar_date >= '{start_date}'
""")

# select * from SNC_cal where calendar_date >= '{start_date}'

# COMMAND ----------

# Primary keys have to be integer type (not decimal)
sdf = sdf.withColumn("USER_ID", sdf["USER_ID"].cast('string')).withColumn("CALENDAR_DATE",sdf["CALENDAR_DATE"].cast('timestamp'))


# COMMAND ----------

# MAGIC %md ## Name Feature table

# COMMAND ----------

# Create a unique table name for each run. This prevents errors if you run the notebook multiple times.
table_name = f"feature_store.dev.user_snc_ncns_calendar"
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

spark.sql(f"""DROP TABLE  {table_name}""")

# COMMAND ----------

fe.create_table(
    name=table_name,
    primary_keys=["USER_ID", "CALENDAR_DATE"],
    timestamp_keys="CALENDAR_DATE",
    df=sdf,
    schema=sdf.schema,
    description="This contains some calculated features about each user's history on a specific day relative to the previous 7, 90, or 365 days. It computes the snc and ncns on a rolling basis."
)


# fe.write_table(
#     name=table_name,
#     df=sdf,
#     mode='merge'
# )


# COMMAND ----------

