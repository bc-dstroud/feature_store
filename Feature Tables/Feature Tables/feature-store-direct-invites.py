# Databricks notebook source
# MAGIC %md # Feature Table for Direct Invites
# MAGIC
# MAGIC - List of Job IDs and User IDs that are direct invites to jobs.
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

# MAGIC %md ## Load dataset
# MAGIC The code in the following cell loads the dataset and does some minor data preparation: creates a unique ID for each observation and removes spaces from the column names. The unique ID column (`wine_id`) is the primary key of the feature table and is used to lookup features.

# COMMAND ----------


sdf = spark.sql(f"""
    /*  Title: Direct Invites
        Date: 1/2/2023
        Author: Dan Streeter
        Summary: 
            In order to better plan overfills, we know the proportion of workers that have successfully worked jobs out of the ones that were still in a successful                applied status at the time of job start.  This logic comes from the data team Time to Fill Logic 
        Ticket Information:
        Dependencies:
            BLUECREW.MYSQL_BLUECREW.JOBS_USERS
        Caveats and Notes:
            This doesn't check the people who worked of the applicants.  It only checks total successful applicants and number successfully worked.
        Modifications:

    */

    --create or replace table PERSONALIZATION.JOB_APPLICATION_CANCELLATION_WORKED as (
    select job_id, user_id, min(1) as direct_invite
    from bc_foreign_mysql.bluecrew.jobs_users
    where user_id is not null and job_id is not null
    group by 1,2
""")



sdf = sdf.withColumn("USER_ID",  sdf["USER_ID"].cast('string')).withColumn("JOB_ID",  sdf["JOB_ID"].cast('string')).withColumn("direct_invite",  sdf["direct_invite"].cast('int'))
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

from pyspark.sql import DataFrame

def write_spark_table_to_databricks_schema(df: DataFrame, table_name: str, schema_name: str = 'bluecrew.ml', mode: str = 'overwrite'):
    """
    Write a Spark DataFrame to a table within a specific schema in Databricks.

    Parameters:
    - df: The Spark DataFrame to write.
    - table_name: The name of the target table.
    - schema_name: The name of the schema (database) in Databricks. Default is 'bluecrew.ml'.
    - mode: Specifies the behavior when the table already exists. Options include:
      - 'append': Add the data to the existing table.
      - 'overwrite': Overwrite the existing table.
      - 'ignore': Silently ignore this operation if the table already exists.
      - 'error' or 'errorifexists': Throw an exception if the table already exists.
    """
    # Define the full table path including the schema name
    full_table_name = f"{schema_name}.{table_name}"
    
    # Write the DataFrame to the table in the specified schema
    df.write.mode(mode).saveAsTable(full_table_name)

    print(f"DataFrame written to table {full_table_name} in mode '{mode}'.")



write_spark_table_to_databricks_schema(optimize_spark(sdf), 'user_work_history', 'bluecrew.default')
sdf = spark.read.format("delta").table('bluecrew.default.user_work_history')

# COMMAND ----------

# MAGIC %md ## Name Feature table

# COMMAND ----------

# Create a unique table name for each run. This prevents errors if you run the notebook multiple times.
table_name = f"feature_store.dev.direct_invites"
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
    primary_keys=["USER_ID", "JOB_ID"],
    df=sdf,
    schema=sdf.schema,
    description="This contains a list of user IDs and job IDs that represent people who were direct invites to a given job."
)


# fe.write_table(
#     name=table_name,
#     df=sdf,
#     mode='merge'
# )


# COMMAND ----------

