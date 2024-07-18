# Databricks notebook source
import datetime
import pandas as pd
import numpy as np
from databricks.feature_store import FeatureStoreClient
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql.functions import udf, col, to_date, when
from pyspark.sql.window import Window
from pyspark.sql.types import ArrayType, IntegerType, DateType
from pyspark.sql import functions as F, types as T
from datetime import date, timedelta, datetime

# COMMAND ----------

# Setting up database access
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

# COMMAND ----------

from pyspark.sql import DataFrame
def schedule_query(start_date: str, end_date: str) -> DataFrame:
  sdf = spark.sql(f"""
    WITH ShiftDurations as(
    
    SELECT user_id, shift_start as shift_start_date, shift_end as shift_end_date, HOUR(shift_start) as shift_start_hour, HOUR(shift_end) as shift_end_hour,  DAYOFWEEK(shift_start) as shift_start_day_of_week, DAYOFWEEK(shift_end) as shift_end_day_of_week, segment_id,
    DATEDIFF(HOUR, shift_start, shift_end) AS shift_duration_hours
    from bc_foreign_snowflake.dm.dm_cm_hours_worked 
    WHERE shift_start >= '{start_date}' and shift_start <= '{end_date}'
    ORDER BY user_id, shift_start),
    RankedShifts AS (
      SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY user_id, shift_start_date ORDER BY shift_duration_hours DESC, shift_start_hour) AS rn
      FROM
        ShiftDurations
    )
    -- ,
    -- rn as(
    SELECT
      user_id,
      shift_start_date,
      shift_start_hour,
      shift_end_date,
      shift_end_hour,
      shift_duration_hours,
      shift_start_day_of_week,
      shift_end_day_of_week
    FROM
      RankedShifts
    WHERE
      rn = 1
    """)
  sdf = sdf.withColumn("USER_ID",  sdf["USER_ID"].cast('string'))
  return sdf

# COMMAND ----------

start_date = '2023-01-01'
now = datetime.now()
end_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")

user_schedule_df = schedule_query(start_date,end_date)
display(user_schedule_df)


# COMMAND ----------

# user_schedule_query = '''
# SELECT * FROM personalization.user_schedule_hours
#     '''

# COMMAND ----------

# user_schedule_df= spark.read.format("snowflake").options(**options).option("query", user_schedule_query).load()

# COMMAND ----------

# display(user_schedule_df)

# COMMAND ----------

for col in user_schedule_df.columns:
    user_schedule_df = user_schedule_df.withColumnRenamed(col, col.lower())

# COMMAND ----------

# Convert dates to datetime and hours to integers
user_schedule_df = user_schedule_df.withColumn('shift_start_hour', user_schedule_df.shift_start_hour.cast('int'))\
  .withColumn('shift_end_hour', user_schedule_df.shift_end_hour.cast('int'))\
    .withColumn('shift_start_day_of_week', user_schedule_df.shift_start_day_of_week.cast('int'))\
      .withColumn('shift_duration_hours', user_schedule_df.shift_duration_hours.cast('int'))


# COMMAND ----------

# # Define the UDF to update the schedule array
# @F.udf(returnType=T.ArrayType(T.IntegerType()))
# def update_schedule(shift_start_hour, shift_end_hour):
#     # Create an array for the entire week, initialized to 0
#     schedule = [0] * 168  # 24 hours * 7 days = 168 hours
    
#     # Update schedule for the current day
#     start_idx = shift_start_hour
#     if shift_end_hour > shift_start_hour:
#         end_idx = shift_end_hour
#         schedule[start_idx:end_idx] = [1] * (end_idx - start_idx)
#     else:
#         # Shift goes into the next day
#         schedule[start_idx:24] = [1] * (24 - start_idx)  # Fill to end of day
#         schedule[0:shift_end_hour] = [1] * shift_end_hour  # Fill start of next day
        
#     return schedule

# Define the UDF to update the schedule array
@F.udf(returnType=T.ArrayType(T.IntegerType()))
def update_schedule(shift_start_hour, shift_end_hour, shift_start_day):
    # Create an array for the entire week, initialized to 0
    schedule = [0] * 168  # 24 hours * 7 days = 168 hours
    
    # Update schedule for the current day
    start_idx = (shift_start_day-1)*24+shift_start_hour
    if shift_end_hour > shift_start_hour:
        end_idx = (shift_start_day-1)*24+shift_end_hour
        schedule[start_idx:end_idx] = [1] * (end_idx - start_idx)
    else:
        # Shift goes into the next day
        if shift_start_day <7:
            end_idx = (shift_start_day)*24+shift_end_hour
            schedule[start_idx:end_idx] = [1] * (end_idx - start_idx)
        else:
            end_idx = shift_end_hour
            schedule[start_idx:168] = [1] * (168 - start_idx)  # Fill to end of day
            schedule[0:shift_end_hour] = [1] * shift_end_hour  # Fill start of next day
    
        
    return schedule

# Apply the UDF to each row to get the schedule array for each shift
user_schedule_df = user_schedule_df.withColumn(
    "schedule_array", 
    update_schedule(F.col("shift_start_hour"), F.col("shift_end_hour"), F.col("shift_start_day_of_week"))
)

# Define a window spec for calculating the running sum
window_spec = Window.partitionBy("user_id").orderBy("shift_start_date").rowsBetween(Window.unboundedPreceding, Window.currentRow)

# Calculate the running sum of the schedule array
user_schedule_df = user_schedule_df.withColumn(
    "cumulative_schedule_array",
    F.collect_list("schedule_array").over(window_spec)
)

# Define UDF to flatten and sum the nested arrays
@F.udf(returnType=T.ArrayType(T.IntegerType()))
def flatten_and_sum(arrays):
    # Convert list of lists to a 2D numpy array and sum along the 0th axis to get cumulative sum
    return np.sum(np.array(arrays, dtype=int), axis=0).tolist()

# Apply the UDF to get the flattened cumulative schedule array
user_schedule_df = user_schedule_df.withColumn(
    "flattened_cumulative_schedule",
    flatten_and_sum(F.col("cumulative_schedule_array"))
)

# COMMAND ----------

user_schedule_df = user_schedule_df.withColumnRenamed("shift_start_date", "calendar_date")\
       .withColumnRenamed("flattened_cumulative_schedule", "running_schedule_array")

# COMMAND ----------

user_schedule_df = user_schedule_df.select("user_id", "calendar_date", "running_schedule_array")

# COMMAND ----------

fe = FeatureEngineeringClient()
table_name = 'feature_store.dev.user_schedule_array2'

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS " + table_name)
fe.create_table(
  name= table_name,
  primary_keys=['user_id', 'calendar_date'],
  timestamp_keys=['calendar_date'],
  df = user_schedule_df,
  description='User Schedule Array computed at the completion of each shift worked. It represents the number of times a given CM has worked a given hour of the week (168 total hours in a week).'
)


# fe.write_table(
#     name=table_name,
#     df=sdf,
#     mode='merge'
# )
