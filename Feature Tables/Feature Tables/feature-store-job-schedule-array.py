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

from pyspark.sql import functions as F, types as T
from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql.window import Window
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DateType, BooleanType, IntegerType
from datetime import datetime, timedelta


# COMMAND ----------

now = datetime.now()
start_date = (now - timedelta(days=31)).strftime("%Y-%m-%d")
# start_date = '2023-01-01'
# activity_end_date = (now - timedelta(days=62)).strftime("%Y-%m-%d")

# COMMAND ----------

# MAGIC %md ## Load dataset
# MAGIC The code in the following cell loads the dataset and does some minor data preparation: creates a unique ID for each observation and removes spaces from the column names. The unique ID column (`wine_id`) is the primary key of the feature table and is used to lookup features.

# COMMAND ----------

# /*  Title: JOB_SCHEDULE_ARRAYS
#     Date: 2/5/2024
#     Author: Dan Streeter
#     Summary: 
#         This uses DM.DM_JOB_TIME_SEGMENTS table to build a shift schedule vector for a job_id
#     Ticket Information:
#     Dependencies:
#         BLUECREW.DM.DM_CALENDAR_DAYS 
#         BLUECREW.DM.DM_JOB_TIME_SEGMENTS
#     Caveats and Notes:
#     Modifications:

# */



sdf = spark.sql(f"""
                with first_table as (
select ts.job_id, ts.start_time as shift_start_date, row_number() over(partition by ts.job_id order by ts.start_time) as row_tracker, max(ts.start_time) over(partition by ts.job_id) as max_start,
 HOUR(ts.start_time) as shift_start_hour, HOUR(ts.end_time) as shift_end_hour, DAYOFWEEK(ts.start_time) as shift_start_day, DAYOFWEEK(ts.start_time) as shift_end_day, DATEDIFF(HOUR, ts.start_time, ts.end_time) AS shift_duration_hours, j.job_created_at
from bc_foreign_snowflake.DM.DM_JOB_TIME_SEGMENTS ts
left join bc_foreign_snowflake.DM.dm_jobs j
on ts.job_id = j.job_id
where j.job_created_at >= '{start_date}'
order by ts.job_id, ts.start_time)

select *, max(row_tracker) over(partition by job_id) max_row
from first_table
""")
sdf = sdf.withColumn("job_id",  sdf["job_id"].cast('string'))
display(sdf)



# COMMAND ----------

sdf = sdf.withColumn('shift_start_hour', sdf.shift_start_hour.cast('int'))\
  .withColumn('shift_end_hour', sdf.shift_end_hour.cast('int'))\
    .withColumn('shift_start_day', sdf.shift_start_day.cast('int'))\
      .withColumn('shift_duration_hours', sdf.shift_duration_hours.cast('int'))


# COMMAND ----------

# Define the UDF to update the schedule array
@F.udf(returnType=T.ArrayType(T.IntegerType()))
def update_job_schedule(shift_start_hour, shift_end_hour, shift_start_day):
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
sdf = sdf.withColumn(
    "schedule_array", 
    update_job_schedule(F.col("shift_start_hour"), F.col("shift_end_hour"), F.col("shift_start_day"))
)

# Define a window spec for calculating the running sum
window_spec = Window.partitionBy("job_id").orderBy("shift_start_date").rowsBetween(Window.unboundedPreceding, Window.currentRow)

# Calculate the running sum of the schedule array
sdf = sdf.withColumn(
    "cumulative_schedule_array",
    F.collect_list("schedule_array").over(window_spec)
)

sdf2 = sdf.where(sdf.row_tracker == sdf.max_row)

# Define UDF to flatten and sum the nested arrays
@F.udf(returnType=T.ArrayType(T.IntegerType()))
def flatten_and_sum(arrays):
    # Convert list of lists to a 2D numpy array and sum along the 0th axis to get cumulative sum
    return np.sum(np.array(arrays, dtype=int), axis=0).tolist()

# Apply the UDF to get the flattened cumulative schedule array
sdf2 = sdf2.withColumn(
    "job_schedule",
    flatten_and_sum(F.col("cumulative_schedule_array"))
)

display(sdf2)



# COMMAND ----------

# MAGIC %md ## Name Feature table

# COMMAND ----------

# Create a unique table name for each run. This prevents errors if you run the notebook multiple times.
table_name = f"feature_store.dev.job_schedule_array"
print(table_name)

# COMMAND ----------

# MAGIC %md ## Create the feature table

# COMMAND ----------

fe = FeatureEngineeringClient()



# COMMAND ----------

# MAGIC %md Create the feature table. For a complete API reference, see ([AWS](https://docs.databricks.com/machine-learning/feature-store/python-api.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/feature-store/python-api)|[GCP](https://docs.gcp.databricks.com/machine-learning/feature-store/python-api.html)).

# COMMAND ----------

# spark.sql(f"""DROP TABLE if Exists {table_name}""")

# COMMAND ----------

# fe.create_table(
#     name=table_name,
#     primary_keys=["job_id", "job_created_at"],
#     timestamp_keys="job_created_at",
#     df=sdf2["job_id", "job_created_at", "job_schedule"],
#     description="This contains a length 168 array with the number of shifts a given job has each hour of the week.  It can be used to determine if the schedule of the job is similar to user history or preferences."
# )


fe.write_table(
    name=table_name,
    df=sdf2["job_id", "job_created_at", "job_schedule"],
    mode='merge'
)


# COMMAND ----------

