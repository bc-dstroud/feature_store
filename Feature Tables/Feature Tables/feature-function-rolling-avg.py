# Databricks notebook source
# Specify the catalog and schema to use. You must have USE_CATALOG privilege on the catalog and USE_SCHEMA and CREATE_TABLE privileges on the schema.
# Change the catalog and schema here if necessary.

catalog_name = "feature_store"
schema_name = "dev"

# COMMAND ----------

from databricks import feature_engineering

fe = feature_engineering.FeatureEngineeringClient()
feature_table_name_jobs = f"{catalog_name}.{schema_name}.job_schedule_array"
feature_table_name_users = f"{catalog_name}.{schema_name}.user_schedule_array2"
function_name = f"{catalog_name}.{schema_name}.cosine_similarity"
feature_spec_name = f"{catalog_name}.{schema_name}.test_spec"

# COMMAND ----------

# dataset here
sdf = spark.sql(f"""
        /*  Title: JOB_APPLICATION_CANCELLATION_WORKED
            Date: 1/2/2023
            Author: Dan Streeter
            Summary: 
                Test query to get user and job IDs and user lat/long to run in feature spec

        */
    select jac.cancel_user_id, jac.cancel_job_id, jac.cancel_created_at, jac.user_address_latitude, jac.user_address_longitude, jac.job_type
    from bc_foreign_snowflake.dm.fact_job_assignment_cancellations jac
    where cancel_created_at >= '2024-03-10'
    -- and job_type = 'Event Staff' 
    """)


sdf = sdf.withColumn("cancel_job_id",  sdf["cancel_job_id"].cast('int'))\
    .withColumn("cancel_user_id",  sdf["cancel_user_id"].cast('int'))\
    .withColumn("user_address_latitude",  sdf["user_address_latitude"].cast('float'))\
    .withColumn("user_address_longitude",  sdf["user_address_longitude"].cast('float'))

display(sdf)

# COMMAND ----------

# Define the function. This function calculates the distance between two locations. 
spark.sql(f"""
CREATE OR REPLACE FUNCTION {function_name}(arr1 ARRAY<INT>, arr2 ARRAY<INT>)
RETURNS FLOAT
LANGUAGE PYTHON AS
$$
from scipy.spatial import distance
# Ensure the arrays are not empty to avoid division by zero
if not arr1 or not arr2:
    return None
# Compute cosine similarity
return 1 - distance.cosine(arr1, arr2)
$$
""")

# COMMAND ----------

from databricks.feature_engineering import (
  FeatureFunction,
  FeatureLookup,
  FeatureEngineeringClient,
)

fe = FeatureEngineeringClient()

features = [
  # Lookup column `average_yearly_spend` and `country` from a table in UC by the input `user_id`.
  FeatureLookup(
    table_name=feature_table_name_jobs,
    lookup_key="cancel_job_id",
    timestamp_lookup_key="cancel_created_at",
    # feature_names=["JOB_ADDRESS_LATITUDE", "JOB_ADDRESS_LONGITUDE"]
  ),
  FeatureLookup(
    table_name=feature_table_name_users,
    lookup_key="cancel_user_id",
    timestamp_lookup_key="cancel_created_at",
    # feature_names=["JOB_ADDRESS_LATITUDE", "JOB_ADDRESS_LONGITUDE"]
  ),
  # Calculate a new feature called `spending_gap` - the difference between `ytd_spend` and `average_yearly_spend`.
  FeatureFunction(
    udf_name=function_name,
    output_name="cosine_sim",
    # Bind the function parameter with input from other features or from request.
    # The function calculates a - b.
    input_bindings={"arr1":"job_schedule", "arr2": "running_schedule_array"},
  ),
]

# Create a `FeatureSpec` with the features defined above.
# The `FeatureSpec` can be accessed in Unity Catalog as a function.
# try: 
#   fe.create_feature_spec(name=feature_spec_name, features=features, exclude_columns=None)
# except Exception as e:
#   if "already exists" in str(e):
#     pass
#   else:
#     raise e

training_set = fe.create_training_set(
    df = sdf, # joining the original Dataset, with our FeatureLookupTable
    feature_lookups=features,
    exclude_columns=[],
    label='job_type'
)



# COMMAND ----------


training_pd = training_set.load_df()
display(training_pd)

# COMMAND ----------

