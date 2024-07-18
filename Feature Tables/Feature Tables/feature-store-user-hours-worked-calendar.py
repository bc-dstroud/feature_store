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
from pyspark.sql.functions import explode, count, size, array_distinct


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
start_date = '2018-01-01'
# end_date = '2023-11-01'

startdate = pd.to_datetime(start_date).date()
# enddate = pd.to_datetime(start_date).date()

# COMMAND ----------

# /*  Title: USER_SCHEDULE_WORKED_CALENDAR
#     Date: 2/5/2023
#     Author: Antwoine Flowers & Dan Streeter
#     Summary: 
#         This uses the USER_WORKED_FEATURES table to turn it into a calendar, creating rolling averages of wages, counts of shifts worked, etc
#     Ticket Information:
#     Dependencies:
#         BLUECREW.DM.DM_CALENDAR_DAYS
#         BLUECREW.DM.FACT_HOURS_WORKED
#         BLUECREW.DM.DM_CM_HOURS_WORKED
#     Caveats and Notes:
#     Modifications:

# */
sdf = spark.sql("""
WITH tmp_calendar AS (
  SELECT EXPLODE(SEQUENCE(TO_TIMESTAMP('2020-01-01 00:00:00'), TO_TIMESTAMP('2026-01-01 00:00:00'), INTERVAL 1 HOUR)) AS DT
)

, hourly_calendar as (SELECT
    tc.DT                       AS CALENDAR_HOUR,
    DATE(tc.DT)                 AS CALENDAR_DATE,
    YEAR(tc.DT)                 AS CALENDAR_YEAR,
    QUARTER(tc.DT)              AS CALENDAR_QUARTER,
    MONTH(tc.DT)                AS CALENDAR_MONTH,
    weekofyear(tc.DT)           AS CALENDAR_WEEK,
    dayofweek(tc.DT)            AS CALENDAR_DAY_OF_WEEK,
    day(tc.DT)                  AS CALENDAR_DAY_NAME,
    DAYOFYEAR(tc.DT)            AS CALENDAR_DAY_NUMBER,
    fc.YEAR::INTEGER            AS FINANCE_YEAR,
    fc.PERIOD                   AS FINANCE_PERIOD,
    fc.WEEK_NUMBER::INTEGER     AS FINANCE_WEEK_NUMBER,
    fc.START_DATE::DATE         AS FINANCE_START_DATE,
    fc.END_DATE::DATE           AS FINANCE_END_DATE
FROM tmp_calendar tc
JOIN bc_foreign_snowflake.DM.DM_FINANCE_CALENDAR fc
    ON tc.DT BETWEEN fc.START_DATE AND fc.END_DATE
ORDER BY 1 ASC)

, user_dates as (
select user_id,
    min(shift_start) as min_date,
    max(shift_start) as max_date
    from bc_foreign_snowflake.dm.dm_cm_hours_worked
    -- where shift_start >= '2018-01-01'
    group by 1
)
, hours_worked_calendar as (
SELECT 
    calendar_date, 
    ud.user_id, 
    calendar_week,
    calendar_day_of_week,
    min_date,
    count(distinct segment_id) as shifts_worked, 
    mode(job_id) as jobs_worked,
    sum(hw.hours_worked) as hours_worked,
    avg(hw.job_wage) as avg_wage
FROM hourly_calendar 
INNER JOIN USER_DATES ud
-- ON calendar_date >=date_trunc('DAY',min_date)
ON calendar_date >min_date
AND calendar_date <= dateadd(max_date, 90)
and calendar_date <= current_date()
LEFT JOIN bc_foreign_snowflake.DM.FACT_HOURS_WORKED hw
on DATE_TRUNC('DAY', hw.SHIFT_start) = date_add(calendar_date,-1)
AND hw.user_id = ud.user_id

group by 1, 2, 3, 4, 5
order by 2, 1)

,
test as (
    select calendar_date,
    CAST(user_id AS STRING) AS user_id,
    min_date as first_worked,
    -- shifts_worked as shifts_worked_yesterday,
    -- jobs_worked as jobs_worked_yesterday,
    -- hours_worked as hours_worked_yesterday,
    -- avg_wage as average_wage_yesterday,
    array_agg(jobs_worked) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as jobs_worked_total,
    array_agg(jobs_worked) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN 30 PRECEDING AND CURRENT ROW) as jobs_worked_last_30_days,
    array_agg(jobs_worked) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN 90 PRECEDING AND CURRENT ROW) as jobs_worked_last_90_days,
    -- SUM(jobs_worked) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN 365 PRECEDING AND CURRENT ROW) as jobs_worked_last_365_days,
    
    SUM(shifts_worked) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as shifts_worked_total,
    SUM(shifts_worked) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) as shifts_worked_last_7_days,
    SUM(shifts_worked) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN 30 PRECEDING AND CURRENT ROW) as shifts_worked_last_30_days,
    SUM(shifts_worked) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN 90 PRECEDING AND CURRENT ROW) as shifts_worked_last_90_days,
    -- SUM(shifts_worked) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN 365 PRECEDING AND CURRENT ROW) as shifts_worked_last_365_days,
    
    -- SUM(hours_worked) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as total_hours_worked_total,
    --SUM(hours_worked) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) as total_hours_worked_last_7_days,
    -- SUM(hours_worked) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN 30 PRECEDING AND CURRENT ROW) as total_hours_worked_last_30_days,
    -- SUM(hours_worked) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN 90 PRECEDING AND CURRENT ROW) as total_hours_worked_last_90_days,
    -- SUM(hours_worked) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN 365 PRECEDING AND CURRENT ROW) as total_hours_worked_last_365_days,
    
    -- AVG(hours_worked) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as avg_hours_worked_total,
    -- AVG(hours_worked) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) as avg_hours_worked_last_7_days,
    -- AVG(hours_worked) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN 30 PRECEDING AND CURRENT ROW) as avg_hours_worked_last_30_days,
    -- AVG(hours_worked) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN 90 PRECEDING AND CURRENT ROW) as avg_hours_worked_last_90_days,
    -- AVG(hours_worked) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN 365 PRECEDING AND CURRENT ROW) as avg_hours_worked_last_365_days,
    
    AVG(avg_wage) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as avg_wage_total,
    AVG(avg_wage) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) as avg_wage_last_7_days,
    AVG(avg_wage) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN 30 PRECEDING AND CURRENT ROW) as avg_wage_last_30_days,
    AVG(avg_wage) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN 90 PRECEDING AND CURRENT ROW) as avg_wage_last_90_days
    
    --,
    --AVG(avg_wage) OVER (PARTITION BY user_id ORDER BY calendar_date ROWS BETWEEN 365 PRECEDING AND CURRENT ROW) as avg_wage_last_365_days
    
from hours_worked_calendar
order by user_id desc, calendar_date)

select * from test
where calendar_date >= '2022-01-01'
""")


# COMMAND ----------

# Primary keys have to be integer type (not decimal)
sdf = sdf.withColumn("user_id", sdf["user_id"].cast('string')).withColumn("calendar_date",sdf["calendar_date"].cast('timestamp'))


# COMMAND ----------

sdf = sdf.withColumn("jobs_worked_last_30_days", size(array_distinct("jobs_worked_last_30_days")))\
  .withColumn("jobs_worked_last_90_days", size(array_distinct("jobs_worked_last_90_days")))\
    .withColumn("jobs_worked_total", size(array_distinct("jobs_worked_total")))
# display(sdf)

# COMMAND ----------

# MAGIC %md ## Name Feature table

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
sdf = optimize_spark(sdf)


# COMMAND ----------

# %sql
# drop table if exists bluecrew.default.user_hours

# COMMAND ----------


# from pyspark.sql import DataFrame

# def write_spark_table_to_databricks_schema(df: DataFrame, table_name: str, schema_name: str = 'bluecrew.ml', mode: str = 'overwrite'):
#     """
#     Write a Spark DataFrame to a table within a specific schema in Databricks.

#     Parameters:
#     - df: The Spark DataFrame to write.
#     - table_name: The name of the target table.
#     - schema_name: The name of the schema (database) in Databricks. Default is 'bluecrew.ml'.
#     - mode: Specifies the behavior when the table already exists. Options include:
#       - 'append': Add the data to the existing table.
#       - 'overwrite': Overwrite the existing table.
#       - 'ignore': Silently ignore this operation if the table already exists.
#       - 'error' or 'errorifexists': Throw an exception if the table already exists.
#     """
#     # Define the full table path including the schema name
#     full_table_name = f"{schema_name}.{table_name}"
    
#     # Write the DataFrame to the table in the specified schema
#     df.write.mode(mode).saveAsTable(full_table_name)

#     print(f"DataFrame written to table {full_table_name} in mode '{mode}'.")



# write_spark_table_to_databricks_schema(optimize_spark(sdf), 'user_hours', 'bluecrew.default')
# sdf = spark.read.format("delta").table('bluecrew.default.user_hours')

# COMMAND ----------

# Create a unique table name for each run. This prevents errors if you run the notebook multiple times.
table_name = f"feature_store.dev.user_hours_worked_calendar2"
print(table_name)

# COMMAND ----------

# MAGIC %md ## Create the feature table

# COMMAND ----------

# MAGIC %md Create the feature table. For a complete API reference, see ([AWS](https://docs.databricks.com/machine-learning/feature-store/python-api.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/feature-store/python-api)|[GCP](https://docs.gcp.databricks.com/machine-learning/feature-store/python-api.html)).

# COMMAND ----------

spark.sql(f"""DROP TABLE IF EXISTS {table_name}""")

# COMMAND ----------


fe = FeatureEngineeringClient()


fe.create_table(
    name=table_name,
    primary_keys=["user_id", "calendar_date"],
    timestamp_keys="calendar_date",
    df=sdf,
    schema=sdf.schema,
    description="This contains some calculated features about each user's history on a specific day relative to the previous 7, or 90 days. It computes the average shifts worked, hours worked, jobs worked, and wage for these timeframes."
)


# fe.write_table(
#     name=table_name,
#     df=sdf,
#     mode='merge'
# )


# COMMAND ----------

