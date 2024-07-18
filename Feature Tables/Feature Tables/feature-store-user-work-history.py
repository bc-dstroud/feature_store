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
# start_date = (now - timedelta(days=31)).strftime("%Y-%m-%d")
start_date = '2023-01-01'
# end_date = '2024-02-01'
end_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")

# COMMAND ----------

# MAGIC %md ## Load dataset
# MAGIC The code in the following cell loads the dataset and does some minor data preparation: creates a unique ID for each observation and removes spaces from the column names. The unique ID column (`wine_id`) is the primary key of the feature table and is used to lookup features.

# COMMAND ----------


sdf = spark.sql(f"""
    /*  Title: JOB_APPLICATION_CANCELLATION_WORKED
        Date: 1/2/2023
        Author: Dan Streeter
        Summary: 
            In order to better plan overfills, we know the proportion of workers that have successfully worked jobs out of the ones that were still in a successful                applied status at the time of job start.  This logic comes from the data team Time to Fill Logic 
        Ticket Information:
        Dependencies:
            BLUECREW.MYSQL_BLUECREW.TIME_SEGMENTS_ABSOLUTE
            BLUECREW.MYSQL_BLUECREW.SCHEDULE_WORK_REQUESTS
            BLUECREW.DM.DM_JOB_NEEDED_HISTORY
            BLUECREW.DM.DM_JOBS
            BLUECREW.DM.DM_CM_JOB_APPLIED_HISTORY
            DM.DM_CM_HOURS_WORKED
        Caveats and Notes:
            This doesn't check the people who worked of the applicants.  It only checks total successful applicants and number successfully worked.
        Modifications:

    */

    --create or replace table PERSONALIZATION.JOB_APPLICATION_CANCELLATION_WORKED as (
    WITH tmp_first_shift AS (
        -- identify the first shift of a job
        SELECT DISTINCT
            tsa.JOB_ID,
            FIRST_VALUE(s.SCHEDULE_ID) OVER (PARTITION BY s.WORK_REQUEST_ID ORDER BY s.CREATED_AT DESC) AS SCHEDULE_ID,
            FIRST_VALUE(tsa.START_TIME) OVER (PARTITION BY tsa.JOB_ID ORDER BY tsa.START_TIME ASC) AS FIRST_SHIFT_START_TIME,
            FIRST_VALUE(tsa.SEGMENT_INDEX) OVER (PARTITION BY tsa.JOB_ID ORDER BY tsa.START_TIME ASC) AS FIRST_SEGMENT
        FROM bc_foreign_mysql.bluecrew.time_segments_absolute tsa
        LEFT JOIN bc_foreign_mysql.bluecrew.SCHEDULE_WORK_REQUESTS s
            ON tsa.JOB_ID = s.WORK_REQUEST_ID
        WHERE tsa.ACTIVE = TRUE
            -- add a shift injects shifts into previous jobs with a matching wage and job type
            -- these retroactive shifts should not be included in time to fill calculations
            AND tsa.CREATED_AT < CAST(FROM_UNIXTIME(tsa.START_TIME) AS TIMESTAMP)
            --AND tsa._FIVETRAN_DELETED = FALSE
            ), -- account for bug causing duplicate rows in TSA

    tmp_first_shift_full AS (
        -- get all columns for first shift and job information
            SELECT
                fs.JOB_ID,
                tsa.SEGMENT_INDEX,
                j.POSITION_ID,
                CAST(FROM_UNIXTIME(tsa.START_TIME) as timestamp) AS START_TIME,
                TIMESTAMPADD(HOUR, -(tsa.START_TIME_OFFSET/60), CAST(FROM_UNIXTIME(tsa.START_TIME)as timestamp)) AS START_TIME_LOCAL,
                CAST(FROM_UNIXTIME(tsa.END_TIME) as timestamp) AS END_TIME,
                TIMESTAMPADD(HOUR, -(tsa.END_TIME_OFFSET/60),CAST(FROM_UNIXTIME(tsa.END_TIME) as timestamp)) AS END_TIME_LOCAL,
                fs.SCHEDULE_ID,
                j.JOB_CREATED_AT,
                j.JOB_TYPE,
                j.JOB_TITLE,
                j.JOB_WAGE,
                j.JOB_STATUS_ENUM,
                j.JOB_STATUS,
                j.JOB_OVERFILL,
                j.JOB_CITY,
                j.JOB_STATE,
                j.COMPANY_ID,
                j.INVITED_WORKER_COUNT,
                j.JOB_NEEDED_ORIGINAL_COUNT,
                COALESCE(nh.NEEDED, j.JOB_NEEDED_ORIGINAL_COUNT) AS NEEDED,
                j.JOB_SHIFTS
                /*
                j.EXTERNAL_JOB_ID,
                j.JOB_TEMPLATE_ID,
                j.JOB_TEMPLATES_EXTERNAL_ID,
                j.BLUESHIFT_REQUEST_ID,
                j.POSITION_ID,
                j.JOB_DESCRIPTION,
                j.JOB_ADDRESS_ID,
                j.JOB_SUPERVISOR_USER_ID,
                j.JOB_POSTER_ID,
                j.JOB_TYPE_ID,
                j.JOB_START_DATE_TIME,
                j.JOB_END_DATE_TIME,
                j.JOB_START_DATE_TIME_LOCAL,
                j.JOB_END_DATE_TIME_LOCAL,
                j.JOB_TIMEZONE,
                j.UPDATED_AT,
                j.JOB_NEEDED_LAST_COUNT,
                j.JOB_BATCH_SIZE,
                j.JOB_DAYS,
                j.JOB_SHIFTS,
                j.JOB_REASON_CODE,
                j.JOB_REASON_TEXT,
                j.JOB_ADDRESS,
                j.JOB_ADDRESS_LINE_TWO,
                j.JOB_ZIPCODE,
                j.JOB_ADDRESS_LATITUDE,
                j.JOB_ADDRESS_LONGITUDE
                */
            FROM tmp_first_shift fs
            LEFT JOIN bc_foreign_mysql.bluecrew.TIME_SEGMENTS_ABSOLUTE tsa
                ON tsa.JOB_ID = fs.JOB_ID
                AND tsa.START_TIME = fs.FIRST_SHIFT_START_TIME
                AND tsa.SEGMENT_INDEX = fs.FIRST_SEGMENT
                AND tsa.ACTIVE = TRUE
                --AND tsa._FIVETRAN_DELETED = FALSE -- account for duplicate rows bug
            LEFT JOIN bc_foreign_snowflake.DM.DM_JOB_NEEDED_HISTORY nh
                ON nh.JOB_ID = fs.JOB_ID
                AND CAST(FROM_UNIXTIME(START_TIME) as timestamp) BETWEEN START_DATE AND END_DATE
            INNER JOIN bc_foreign_snowflake.DM.DM_JOBS j
                ON j.JOB_ID = fs.JOB_ID
    --                 AND JOB_STATUS_ENUM < 6 -- active jobs only
            WHERE YEAR(CAST(FROM_UNIXTIME(START_TIME) as timestamp)) >= 2020
                AND CAST(FROM_UNIXTIME(START_TIME) as timestamp) <= DATEADD(DAY, 28, CURRENT_DATE())
                AND (j.INVITED_WORKER_COUNT IS NULL OR j.INVITED_WORKER_COUNT < COALESCE(nh.NEEDED, j.JOB_NEEDED_ORIGINAL_COUNT))
            )
            ,
            successful_applications as (
                SELECT user_id, job_id, min(jah.APPLIED_STATUS_START_DATE) as min_successful_app_start, max(jah.APPLIED_STATUS_START_DATE) as max_successful_app_start, min(jah.END_DATE) as min_successful_app_end, max(jah.END_DATE) as max_successful_app_end, count(applied_status_enum) as successful_application_count
                FROM bc_foreign_snowflake.DM.DM_CM_JOB_APPLIED_HISTORY jah 
                where APPLIED_STATUS_ENUM = 0
                group by 1,2
            )
            ,
            successful_job_applications AS (
                -- Successful applications as of first shift start time
                SELECT
                    fs.JOB_ID,
                    jah.USER_ID,
                    jah.successful_application_count,
                    jah.min_successful_app_start,
                    jah.max_successful_app_start,
                    jah.min_successful_app_end,
                    jah.max_successful_app_end,
                    fs.start_time as JOB_START_TIME,
                    case 
                    when jah.max_successful_app_end>=DATEADD(HOUR,-16,fs.start_time) and jah.max_successful_app_end<fs.start_time then "SNC" 
                    when jah.max_successful_app_end<DATEADD(HOUR,-16,fs.start_time) then "Early Cancel" 
                    when jah.max_successful_app_end>=fs.start_time then "No Cancel" 
                    end
                    as application_status,
                    DATEDIFF(HOUR,jah.min_successful_app_start,fs.start_time) as apply_lead_time_hours
                    --DENSE_RANK() OVER (PARTITION BY jah.JOB_ID ORDER BY jah.APPLIED_STATUS_START_DATE ASC, USER_ID) AS SIGN_UP_ORDER
                FROM tmp_first_shift_full fs
                LEFT JOIN successful_applications jah
                    ON fs.JOB_ID = jah.JOB_ID
                    AND fs.START_TIME >= jah.min_successful_app_start
                where jah.user_id is not null
                )
        ,
            worked as 
            (
                SELECT sja.*, case when (sja.user_id, sja.job_id) in (select hw.user_id, hw.job_id from bc_foreign_snowflake.dm.dm_cm_hours_worked hw) then 'Worked' else application_status end as target_var
                FROM successful_job_applications sja 
            )
            ,
            jacw as (
            select 
            fs.POSITION_ID,
            fs.SEGMENT_INDEX,
            -- fs.START_TIME,
            fs.START_TIME_LOCAL,
            fs.END_TIME,
            fs.END_TIME_LOCAL,
            fs.SCHEDULE_ID,
            fs.JOB_CREATED_AT,
            fs.JOB_TYPE,
            fs.JOB_TITLE,
        --             fs.JOB_DESCRIPTION,
            fs.JOB_NEEDED_ORIGINAL_COUNT,
            fs.JOB_WAGE,
            fs.JOB_STATUS_ENUM,
            fs.JOB_STATUS,
            fs.JOB_OVERFILL,
            fs.JOB_CITY,
            fs.JOB_STATE,
            fs.COMPANY_ID,
            fs.INVITED_WORKER_COUNT,
            fs.NEEDED,
            fs.JOB_SHIFTS,
            w.* from worked w
            LEFT JOIN tmp_first_shift_full fs
            ON fs.JOB_ID = w.JOB_ID
            )
            ,
            sample_output as (
                select user_id, job_start_time,  target_var,
                count(target_var) over(partition by user_id, target_var order by job_start_time) as target_tracker,
                count(target_var) over(partition by user_id order by job_start_time) as total_tracker,
                sum(case when target_var = 'Worked' then 1 else 0 end) over(partition by user_id order by job_start_time) as worked_count,
                sum(case when target_var = 'Early Cancel' then 1 else 0 end) over(partition by user_id order by job_start_time) as early_cancel_count,
                sum(case when target_var = 'No Cancel' then 1 else 0 end) over(partition by user_id order by job_start_time) as no_cancel_count,
                sum(case when target_var = 'SNC' then 1 else 0 end) over(partition by user_id order by job_start_time) as SNC_count,
                row_number() over(partition by user_id, job_start_time order by job_start_time) as row_tracker,
                max(row_number() over(partition by user_id, job_start_time order by job_start_time)) over(partition by user_id, job_start_time) as max_row_tracker
                from jacw
                where 1=1
                -- and job_type = 'Event Staff' 
                --and jacw.JOB_START_TIME >= '{start_date}'
                and jacw.START_TIME_LOCAL < '{end_date}'
            )
            select
            user_id,
            job_start_time,
            total_tracker, 
            worked_count/total_tracker as worked_ratio_all,
            early_cancel_count/total_tracker as early_cancel_ratio_all,
            no_cancel_count/total_tracker as no_cancel_ratio_all,
            SNC_count/total_tracker as SNC_ratio_all,
            worked_count/(worked_count+no_cancel_count) as worked_ratio_ncns,
            worked_count/(worked_count+no_cancel_count+SNC_count) as worked_ratio_SNC
            from sample_output
            where row_tracker=max_row_tracker
            order by user_id, job_start_time

            
""")
# select * from 
#             output_before_limits
#             where job_start_time >='{start_date}'


sdf = sdf.withColumn("USER_ID",  sdf["USER_ID"].cast('string'))
display(sdf)


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
# for x in sdf.columns:
#     sdf = sdf.withColumn(x, F.upper(F.col(x)))
sdf = optimize_spark(sdf)

for col in sdf.columns:
    sdf = sdf.withColumnRenamed(col, col.upper())


# COMMAND ----------

# MAGIC %md ## Name Feature table

# COMMAND ----------

# Create a unique table name for each run. This prevents errors if you run the notebook multiple times.
table_name = f"feature_store.dev.user_work_history"
print(table_name)

# COMMAND ----------

# MAGIC %md ## Create the feature table

# COMMAND ----------

fe = FeatureEngineeringClient()



# COMMAND ----------

# MAGIC %md Create the feature table. For a complete API reference, see ([AWS](https://docs.databricks.com/machine-learning/feature-store/python-api.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/feature-store/python-api)|[GCP](https://docs.gcp.databricks.com/machine-learning/feature-store/python-api.html)).

# COMMAND ----------

spark.sql(f"""DROP TABLE if Exists {table_name}""")

# COMMAND ----------

fe.create_table(
    name=table_name,
    primary_keys=["USER_ID", "JOB_START_TIME"],
    timestamp_keys="JOB_START_TIME",
    df=sdf,
    schema=sdf.schema,
    description="This contains a tracker of the application to conclusion for workers.  The conclusion of a given application is either early cancellation, short notice cancellation (SNC), no cancellation (NCNS), or worked.  If an applicant cancels a given job multiple times, it will only count 1 time based on the final application.  If a CM cancels, but later works a job, it will be counted as worked."
)


# fe.write_table(
#     name=table_name,
#     df=sdf,
#     mode='merge'
# )


# COMMAND ----------

# MAGIC %sql
# MAGIC select * from feature_store.dev.user_work_history
# MAGIC where user_id = 47502

# COMMAND ----------

