# Databricks notebook source
# Specify the catalog and schema to use. You must have USE_CATALOG privilege on the catalog and USE_SCHEMA and CREATE_TABLE privileges on the schema.
# Change the catalog and schema here if necessary.

catalog_name = "feature_store"
schema_name = "dev"

# COMMAND ----------

from databricks import feature_engineering

fe = feature_engineering.FeatureEngineeringClient()
function_name = f"{catalog_name}.{schema_name}.double_division"

# COMMAND ----------

# Define the function. This function calculates the distance between two locations. 
spark.sql(f"""
CREATE OR REPLACE FUNCTION {function_name}(numerator DOUBLE, denominator DOUBLE)
RETURNS FLOAT
LANGUAGE PYTHON AS
$$
# Ensure the numerator and denominator exist and avoid division by zero
if not numerator or not denominator or denominator == 0:
    return None
# Compute cosine similarity
return numerator/denominator
$$
""")

# COMMAND ----------

