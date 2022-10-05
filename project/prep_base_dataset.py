# Databricks notebook source
# %pip install azureml-sdk -q

# COMMAND ----------
import os
import re
import time
import sys
import string
import unicodedata
import pandas as pd
import numpy as np
import datetime
import argparse
from datetime import date

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.context import SparkContext

import azureml.core
from azureml.core import Workspace, Environment, Experiment, Datastore, Dataset, ScriptRunConfig
from azureml.core.datastore import Datastore
from azureml.data.data_reference import DataReference
from azureml.core.authentication import ServicePrincipalAuthentication

def populate_environ():
    parser = argparse.ArgumentParser(description='Process arguments passed to script')

    # The AZUREML_SCRIPT_DIRECTORY_NAME argument will be filled in if the DatabricksStep
    # was run using a local source_directory and python_script_name
    parser.add_argument('--AZUREML_SCRIPT_DIRECTORY_NAME')

    # Remaining arguments are filled in for all databricks jobs and can be used to build the run context
    parser.add_argument('--AZUREML_RUN_TOKEN')
    parser.add_argument('--AZUREML_RUN_TOKEN_EXPIRY')
    parser.add_argument('--AZUREML_RUN_ID')
    parser.add_argument('--AZUREML_ARM_SUBSCRIPTION')
    parser.add_argument('--AZUREML_ARM_RESOURCEGROUP')
    parser.add_argument('--AZUREML_ARM_WORKSPACE_NAME')
    parser.add_argument('--AZUREML_ARM_PROJECT_NAME')
    parser.add_argument('--AZUREML_SERVICE_ENDPOINT')
    parser.add_argument('--AZUREML_WORKSPACE_ID')
    parser.add_argument('--AZUREML_EXPERIMENT_ID')

    parser.add_argument("--base_file_name", type=str, default="ecd_tickets_cleaned_2_more_withJune2022", help="base csv file")

    # parser.add_argument("--output_train", type=str, help="output train path")
    # parser.add_argument("--output_validation", type=str, help="output validation path")
    # parser.add_argument("--output_test", type=str, help="output test path")
    # parser.add_argument("--output_temporal_test", type=str, help="output temporal_test path")

    # parser.add_argument("--input_base_dataset_name", type=str, help="input base dataset")
    # parser.add_argument("--input_base_dataset_version", type=str, help="input base dataset version")
    
    (args, extra_args) = parser.parse_known_args()
    os.environ['AZUREML_RUN_TOKEN'] = args.AZUREML_RUN_TOKEN
    os.environ['AZUREML_RUN_TOKEN_EXPIRY'] = args.AZUREML_RUN_TOKEN_EXPIRY
    os.environ['AZUREML_RUN_ID'] = args.AZUREML_RUN_ID
    os.environ['AZUREML_ARM_SUBSCRIPTION'] = args.AZUREML_ARM_SUBSCRIPTION
    os.environ['AZUREML_ARM_RESOURCEGROUP'] = args.AZUREML_ARM_RESOURCEGROUP
    os.environ['AZUREML_ARM_WORKSPACE_NAME'] = args.AZUREML_ARM_WORKSPACE_NAME
    os.environ['AZUREML_ARM_PROJECT_NAME'] = args.AZUREML_ARM_PROJECT_NAME
    os.environ['AZUREML_SERVICE_ENDPOINT'] = args.AZUREML_SERVICE_ENDPOINT
    os.environ['AZUREML_WORKSPACE_ID'] = args.AZUREML_WORKSPACE_ID
    os.environ['AZUREML_EXPERIMENT_ID'] = args.AZUREML_EXPERIMENT_ID
    return args, extra_args

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()

# base_file_name = "ecd_tickets_cleaned_2_more_withJune2022"
# base_file_name = "ecd_tickets_cleaned_2_more_withJuneJuly2022"
base_file_name = "ecd_tickets_cleaned_2_more_withNewLongDescs"
args, extra_args = populate_environ()
base_file_name = args.base_file_name

pdf = pd.read_csv(f"/dbfs/FileStore/ecd/{base_file_name}.csv")

# COMMAND ----------

pdf['INTERNAL_PRIORITY'] = pdf['INTERNAL_PRIORITY'].astype(str)
pdf['TICKET_SOURCE'] = pdf['TICKET_SOURCE'].astype(str)
pdf['SELF_SERVICE_SOLUTION_FLAG'] = pdf['SELF_SERVICE_SOLUTION_FLAG'].astype(str)
pdf['ACTUAL_COMPLETION_HRS'] = pdf['ACTUAL_COMPLETION_HRS'].astype(str)
pdf['BUSINESS_COMPLETION_HRS'] = pdf['BUSINESS_COMPLETION_HRS'].astype(str)
if 'CLASS_STRUCTURE_ID' in pdf.columns:
    pdf['CLASS_STRUCTURE_ID'] = pdf['CLASS_STRUCTURE_ID'].astype(str)

# pdf.to_parquet('/dbfs/FileStore/ecd/ecd_tickets_cleaned_2_more.parquet')
pdf.to_parquet(f'/dbfs/FileStore/ecd/{base_file_name}.parquet')

# COMMAND ----------

df = spark.read.parquet(f"dbfs:/FileStore/ecd/{base_file_name}.parquet")

df = df.withColumn("REPORT_DATE_DT", F.to_timestamp(F.col('REPORT_DATE'), 'yyyy-MM-dd HH:mm:ss'))
df = df.withColumn("REPORT_MONTH", F.regexp_replace(F.substring('REPORT_DATE', 0, 7), '-', '').cast(IntegerType()))
display(df)

# COMMAND ----------

df.registerTempTable("df_temp")
df_deduped = sqlContext.sql('''
    SELECT 
        *
    FROM (
        SELECT 
            *,
            ROW_NUMBER() OVER (PARTITION BY ticket_id ORDER BY REPORT_DATE_DT DESC) AS rank
        FROM df_temp
    ) vo WHERE rank = 1
''')
df_deduped.drop('rank').write.mode('overwrite').parquet(f'dbfs:/FileStore/ecd/{base_file_name}_deduped.parquet')


# COMMAND ----------

# df = spark.read.format('delta').load("dbfs:/FileStore/ecd/ecd_tickets_cleaned_2.delta")
df = spark.read.parquet(f"dbfs:/FileStore/ecd/{base_file_name}_deduped.parquet")
print(f'Total records: [{df.count()}]')

