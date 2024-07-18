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
from pyspark.sql import functions as F
from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup
from databricks.feature_engineering import FeatureEngineeringClient


from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DateType, BooleanType, IntegerType
from datetime import datetime, timedelta


# COMMAND ----------

now = datetime.now()
# end_date = '2022-01-01'
# activity_end_date = '2021-12-01'
end_date = (now - timedelta(days=30)).strftime("%Y-%m-%d")
activity_end_date = (now - timedelta(days=61)).strftime("%Y-%m-%d")

# COMMAND ----------

# MAGIC %md ## Load dataset
# MAGIC The code in the following cell loads the dataset and does some minor data preparation: creates a unique ID for each observation and removes spaces from the column names. The unique ID column (`wine_id`) is the primary key of the feature table and is used to lookup features.

# COMMAND ----------

# /*  Title: Job Features 2
#     Date: 12/20/23
#     Author: Dan Streeter
#     Summary: 
#         In order to train a machine learning model to achieve better model performance, we need to perform feature engineering to create relevant job-level attributes         at the time of job creation.  Some of these features will include wage delta from the county average wage for the same job title in that year and month,               eligible users in the same county at the time of job creation, active users in the last 7 days in the same county at the time of job creation.
#         One that I would still like to add is a distance measurement showing number of CMs that live within X miles of the job lat/long.  Right now I am limiting all          dates to 1 Jan 23 and later.  If the query is more efficient, I would like to run it for longer
#     Ticket Information:
        
#     Dependencies:
#         IOS_APP.IDENTIFIES
#         PROD_ANDROID.IDENTIFIES
#         DM.DM_USERS_DATA
#         DM.DM_CM_USER_STATUS_HISTORY
#         DM.DM_JOBS,
#         DM.DM_ZIPCODE_LOOKUP
#         DM.DM_CM_
#     Caveats and Notes:
#         This is currently only built for event work job types from 1 Jan 2023 to 1 Nov 2023.
#     Modifications:

# */
# SELECT * FROM bluecrew.product_analytics.job_cancellation;
# update_from_date = '2023-01-01'
# need to set a variable for the update date

# create or replace table bluecrew.personalization.job_data as (
sdf = spark.sql(f"""with activity as (
SELECT distinct internal_id
    , date_trunc('DAY', ORIGINAL_TIMESTAMP) as app_session
FROM bc_foreign_snowflake.ios_app.identifies
WHERE ORIGINAL_TIMESTAMP >= '{activity_end_date}' --AND ORIGINAL_TIMESTAMP < '2023-12-01'
UNION
SELECT distinct internal_id
    , date_trunc('DAY', ORIGINAL_TIMESTAMP) as app_session
FROM bc_foreign_snowflake.prod_android.identifies
WHERE ORIGINAL_TIMESTAMP >= '{activity_end_date}' --AND ORIGINAL_TIMESTAMP < '2023-12-01'
),
-- Maybe group by county and year/month first?
user_status as (
select u.user_id, 
    u.user_zipcode, 
    uh.start_date, 
    uh.end_date, 
    uh.user_status_enum, 
    u.user_address_latitude,
    u.user_address_longitude,
    z.county
from bc_foreign_snowflake.dm.dm_users_data u
left join bc_foreign_snowflake.dm.dm_cm_user_status_history uh
on u.user_id = uh.user_id
left join bc_foreign_snowflake.dm.dm_zip_code_lookup z
on u.user_zipcode = z.zip
where uh.user_status_enum !=0
)
, jobs as (
select avg(job_wage) over(partition by county, job_type, job_title, year(job_created_at), month(job_created_at)) as county_job_type_title_avg_wage,
        -- count(distinct job_id) over(partition by county, job_type, job_title, year(job_created_at), month(job_created_at)) as job_type_title_count,
        DENSE_RANK() OVER (PARTITION BY county, job_type, job_title, year(job_created_at), month(job_created_at) ORDER BY job_id ASC ) +
        DENSE_RANK() OVER (PARTITION BY county, job_type, job_title, year(job_created_at), month(job_created_at) ORDER BY job_id DESC) - 1 AS job_type_title_count,
        -- count(distinct job_id) over(partition by county, year(job_created_at), month(job_created_at)) as total_job_count,
        DENSE_RANK() OVER (PARTITION BY county, year(job_created_at), month(job_created_at) ORDER BY job_id ASC ) +
        DENSE_RANK() OVER (PARTITION BY county, year(job_created_at), month(job_created_at) ORDER BY job_id DESC) - 1 AS total_job_count,
        sum(job_needed_last_count) over(partition by county, year(job_created_at), month(job_created_at)) as total_CMs_required,
    job_wage - avg(job_wage) over(partition by county, job_type, job_title, year(job_created_at), month(job_created_at)) as wage_delta,
    case when avg(job_wage) over(partition by county, job_type, job_title, year(job_created_at), month(job_created_at)) >0 then
        (job_wage/avg(job_wage) over(partition by county, job_type, job_title, year(job_created_at), month(job_created_at)) - 1)*100 
        else 0 
    end as wage_delta_percent,
    job_id,
    job_address_latitude,
    job_address_longitude,
    job_type, 
    job_start_date_time, 
    job_created_at,
    job_wage, 
    job_title, 
    schedule_name, 
    job_needed_last_count, 
    job_overfill, 
    job_shifts, 
    COALESCE(invited_worker_count,0) as invited_worker_count, 
    job_is_application,
    case when (contains(lower(schedule_name), '1st') or contains(lower(schedule_name),'first')) then '1st shift'
        when (contains(lower(schedule_name), '2nd') or contains(lower(schedule_name),'second')) then '2nd shift'
        when (contains(lower(schedule_name), '3rd') or contains(lower(schedule_name),'third')) then '3rd shift'
        when (contains(lower(schedule_name), 'am shift') or contains(lower(schedule_name),'morning')) then 'morning'
        when (contains(lower(schedule_name), 'night') or contains(lower(schedule_name),'pm')) then 'night'
        when (contains(lower(schedule_name), 'mon-fri') or contains(lower(schedule_name),'mon - fri')) then 'mon-fri'
        else 'other'
    end as schedule_name_updated,
    timediff(DAY, job_created_at,job_start_date_time) as posting_lead_time_days,
    z.county
    from bc_foreign_snowflake.dm.dm_jobs j
    left join bc_foreign_snowflake.dm.dm_zip_code_lookup z
    on j.job_zipcode = z.zip
    where j.job_created_at >='{end_date}'
    order by 1 desc, 2, 3, 4)
, 
user_job_statuses as (
select j.*, u.user_id, u.user_status_enum, 
3959 * 2 * ASIN(SQRT(POWER(SIN((user_address_latitude
        - job_address_latitude) * PI() / 180 / 2), 2) + COS(user_address_latitude * PI() / 180) * COS(job_address_latitude * PI() / 180) * POWER(SIN((user_address_longitude - job_address_longitude) * PI() / 180 / 2), 2))) AS distance_miles
from jobs j
left join user_status u
on j.county = u.county
AND u.start_date < j.job_created_at 
AND u.end_date >= j.job_created_at
)
select u.county_job_type_title_avg_wage, 
    u.job_type_title_count,
    u.total_job_count,
    u.total_CMs_required,
    u.wage_delta,
    u.wage_delta_percent,
    u.job_id,
    u.job_type, 
    u.job_address_latitude,
    u.job_address_longitude,
    u.job_start_date_time, 
    u.JOB_CREATED_AT,
    u.job_wage, 
    u.job_title, 
    u.schedule_name, 
    u.job_needed_last_count, 
    u.job_overfill, 
    u.job_shifts, 
    u.invited_worker_count, 
    u.job_is_application,
    u.schedule_name_updated,
    u.posting_lead_time_days,
    u.county,
    count(distinct case when user_status_enum = 1 then u.user_id else null end) as eligible_users,
    count(distinct a.internal_id) as active_users_7_days,
    count(distinct case when distance_miles <= 1 and user_status_enum = 1 then u.user_id else null end) as eligible_CMs_1_mile,
    count(distinct case when distance_miles <= 5 and user_status_enum = 1 then u.user_id else null end) as eligible_CMs_5_mile,
    count(distinct case when distance_miles <= 10 and user_status_enum = 1 then u.user_id else null end) as eligible_CMs_10_mile,
    count(distinct case when distance_miles <= 15 and user_status_enum = 1 then u.user_id else null end) as eligible_CMs_15_mile,
    count(distinct case when distance_miles <= 1 then a.internal_id else null end) as active_CMs_1_mile,
    count(distinct case when distance_miles <= 5 then a.internal_id else null end) as active_CMs_5_mile,
    count(distinct case when distance_miles <= 10 then a.internal_id else null end) as active_CMs_10_mile,
    count(distinct case when distance_miles <= 15 then a.internal_id else null end) as active_CMs_15_mile
from user_job_statuses u
left join activity a
on u.user_id = a.internal_id
AND u.job_created_at >= a.app_session
and TIMEDIFF(DAY, u.job_created_at, a.app_session) > -7
group by 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23""")
# );


# COMMAND ----------

from pyspark.sql.functions import floor, col
from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from pyspark.sql.types import LongType, DecimalType, FloatType
# Table Conversion Util Functions

def optimize_spark(spark_df: DataFrame) -> DataFrame:
    """
    Converts all LongType columns in the Spark DataFrame to DecimalType, and then
    converts all DecimalType columns to FloatType for efficiency.
    
    Parameters:
    - spark_df: The input PySpark DataFrame.
    
    Returns:
    - DataFrame: A new DataFrame with LongType columns converted to DecimalType and
                 DecimalType columns converted to FloatType.
    """
    # First, convert LongType columns to DecimalType
    for field in spark_df.schema.fields:
        if isinstance(field.dataType, LongType):
            spark_df = spark_df.withColumn(field.name, col(field.name).cast(DecimalType(38, 0)))
    
    # Next, convert DecimalType columns to FloatType
    for field in spark_df.schema.fields:
        if isinstance(field.dataType, DecimalType):
            spark_df = spark_df.withColumn(field.name, col(field.name).cast(FloatType()))
    
    return spark_df


# COMMAND ----------

# Primary keys have to be integer type (not decimal)
sdf = optimize_spark(sdf)

# COMMAND ----------


for col in sdf.columns:
    sdf = sdf.withColumnRenamed(col, col.upper())
sdf = sdf.withColumn('CM_COUNT_RATIO', sdf.TOTAL_CMS_REQUIRED / sdf.TOTAL_JOB_COUNT)
sdf = sdf.withColumn("JOB_ID", sdf["JOB_ID"].cast('int'))
sdf = sdf.withColumn("JOB_ID", sdf["JOB_ID"].cast('string'))

# COMMAND ----------

# MAGIC %md ## Name Feature table

# COMMAND ----------

# Create a unique table name for each run. This prevents errors if you run the notebook multiple times.
table_name = f"feature_store.dev.jobs_data"
print(table_name)

# COMMAND ----------

# MAGIC %md ## Create the feature table

# COMMAND ----------

fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md Create the feature table. For a complete API reference, see ([AWS](https://docs.databricks.com/machine-learning/feature-store/python-api.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/feature-store/python-api)|[GCP](https://docs.gcp.databricks.com/machine-learning/feature-store/python-api.html)).

# COMMAND ----------

# spark.sql('DROP TABLE if Exists feature_store.dev.jobs_data')

# COMMAND ----------

# fe.create_table(
#     name=table_name,
#     primary_keys=["JOB_ID", "JOB_CREATED_AT"],
#     timestamp_keys="JOB_CREATED_AT",
#     df=sdf,
#     schema=sdf.schema,
#     description="This contains some calculated features about jobs. It compares the average job wage for the region, job type, and job title to each individual job_id. It also contains the number of eligible CMs in the job at the time of the job start date."
# )


fe.write_table(
    name=table_name,
    df=sdf,
    mode='merge'
)


# COMMAND ----------

a = spark.sql("""
                select min(job_start_date_time)
                from feature_store.dev.jobs_data
              """)
display(a)