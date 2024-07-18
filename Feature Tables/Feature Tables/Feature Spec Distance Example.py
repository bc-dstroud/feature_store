# Databricks notebook source
# Specify the catalog and schema to use. You must have USE_CATALOG privilege on the catalog and USE_SCHEMA and CREATE_TABLE privileges on the schema.
# Change the catalog and schema here if necessary.

catalog_name = "feature_store"
schema_name = "dev"

# COMMAND ----------

from databricks import feature_engineering

fe = feature_engineering.FeatureEngineeringClient()
feature_table_name = f"{catalog_name}.{schema_name}.jobs_data"
function_name = f"{catalog_name}.{schema_name}.distance"
feature_spec_name = f"{catalog_name}.{schema_name}.job_distance_spec"

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
CREATE OR REPLACE FUNCTION {function_name}(lat1 FLOAT, lon1 FLOAT, lat2 FLOAT, lon2 FLOAT)
RETURNS FLOAT
LANGUAGE PYTHON AS
$$
import math

if None in (lat1, lon1, lat2, lon2):
    return None

# Radius of the Earth in miles
R = 3956.0
# Convert latitude and longitude from degrees to radians
lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

# Haversine formula
dlat = lat2 - lat1
dlon = lon2 - lon1
a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
c = 2 * math.asin(math.sqrt(a))
distance = R * c

return distance
$$""")

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
    table_name=feature_table_name,
    lookup_key="cancel_job_id",
    timestamp_lookup_key="cancel_created_at",
    feature_names=["JOB_ADDRESS_LATITUDE", "JOB_ADDRESS_LONGITUDE"]
  ),
  # Calculate a new feature called distance.
  FeatureFunction(
    udf_name=function_name,
    output_name="distance",
    # Bind the function parameter with input from other features or from request.
    # The function calculates a - b.
    input_bindings={"latitude":"JOB_ADDRESS_LATITUDE", "longitude": "JOB_ADDRESS_LONGITUDE", "user_latitude" : "user_address_latitude", "user_longitude": "user_address_longitude"},
  ),
]

# Create a `FeatureSpec` with the features defined above.
# The `FeatureSpec` can be accessed in Unity Catalog as a function.
try: 
  fe.create_feature_spec(name=feature_spec_name, features=features, exclude_columns=None)
except Exception as e:
  if "already exists" in str(e):
    pass
  else:
    raise e

#this isn't actually calling FeatureSpec.  This is using a normal feature lookup from above.  
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

