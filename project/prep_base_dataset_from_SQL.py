# Databricks notebook source
# MAGIC %md
# MAGIC We don't need this notebook to connect back to Azure ML. Everything happens within Databricks. Therefore, AzureML auth code has been commented out.
# MAGIC 
# MAGIC The pip installs are also commented out because they won't run from the AzureML pipeline. Instead the dependencies have been passed in via `pypi_libraries=[...]`.
# MAGIC 
# MAGIC The `python -m spacy download` step has been replaced with an API call below.

# COMMAND ----------

# !pip install presidio_analyzer
# !pip install presidio_anonymizer
# !python -m spacy download en_core_web_lg

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

import spacy
spacy.cli.download("en_core_web_lg")

# import azureml.core
# from azureml.core import Workspace, Environment, Experiment, Datastore, Dataset, ScriptRunConfig
# from azureml.core.datastore import Datastore
# from azureml.data.data_reference import DataReference
# from azureml.core.authentication import ServicePrincipalAuthentication

# def populate_environ():
#     parser = argparse.ArgumentParser(description='Process arguments passed to script')

#     # The AZUREML_SCRIPT_DIRECTORY_NAME argument will be filled in if the DatabricksStep
#     # was run using a local source_directory and python_script_name
#     parser.add_argument('--AZUREML_SCRIPT_DIRECTORY_NAME')

#     # Remaining arguments are filled in for all databricks jobs and can be used to build the run context
#     parser.add_argument('--AZUREML_RUN_TOKEN')
#     parser.add_argument('--AZUREML_RUN_TOKEN_EXPIRY')
#     parser.add_argument('--AZUREML_RUN_ID')
#     parser.add_argument('--AZUREML_ARM_SUBSCRIPTION')
#     parser.add_argument('--AZUREML_ARM_RESOURCEGROUP')
#     parser.add_argument('--AZUREML_ARM_WORKSPACE_NAME')
#     parser.add_argument('--AZUREML_ARM_PROJECT_NAME')
#     parser.add_argument('--AZUREML_SERVICE_ENDPOINT')
#     parser.add_argument('--AZUREML_WORKSPACE_ID')
#     parser.add_argument('--AZUREML_EXPERIMENT_ID')

#     parser.add_argument("--base_file_name", type=str, default="ecd_tickets_cleaned_2_more_withJune2022", help="base csv file")

#     # parser.add_argument("--output_train", type=str, help="output train path")
#     # parser.add_argument("--output_validation", type=str, help="output validation path")
#     # parser.add_argument("--output_test", type=str, help="output test path")
#     # parser.add_argument("--output_temporal_test", type=str, help="output temporal_test path")

#     # parser.add_argument("--input_base_dataset_name", type=str, help="input base dataset")
#     # parser.add_argument("--input_base_dataset_version", type=str, help="input base dataset version")
    
#     (args, extra_args) = parser.parse_known_args()
#     os.environ['AZUREML_RUN_TOKEN'] = args.AZUREML_RUN_TOKEN
#     os.environ['AZUREML_RUN_TOKEN_EXPIRY'] = args.AZUREML_RUN_TOKEN_EXPIRY
#     os.environ['AZUREML_RUN_ID'] = args.AZUREML_RUN_ID
#     os.environ['AZUREML_ARM_SUBSCRIPTION'] = args.AZUREML_ARM_SUBSCRIPTION
#     os.environ['AZUREML_ARM_RESOURCEGROUP'] = args.AZUREML_ARM_RESOURCEGROUP
#     os.environ['AZUREML_ARM_WORKSPACE_NAME'] = args.AZUREML_ARM_WORKSPACE_NAME
#     os.environ['AZUREML_ARM_PROJECT_NAME'] = args.AZUREML_ARM_PROJECT_NAME
#     os.environ['AZUREML_SERVICE_ENDPOINT'] = args.AZUREML_SERVICE_ENDPOINT
#     os.environ['AZUREML_WORKSPACE_ID'] = args.AZUREML_WORKSPACE_ID
#     os.environ['AZUREML_EXPERIMENT_ID'] = args.AZUREML_EXPERIMENT_ID
#     return args, extra_args

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load original tickets (back to 2016) from CSV
# MAGIC These long descriptions have been removed from the EDR as of Sept 2022, so we have to refer to the existing CSV to get them.

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()

base_file_name = "ecd_tickets_cleaned_2_more_withNewLongDescs"
# args, extra_args = populate_environ()

# base_file_name = args.base_file_name # We don't want to uncomment this?? Dependent on what is passed from pipeline (Don't change pipeline!)

pdf = pd.read_csv(f"/dbfs/FileStore/ecd/{base_file_name}.csv")

# COMMAND ----------

last_date = pdf['REPORT_DATE'].max()
last_date = str(last_date).split()[0]
print(last_date)

# COMMAND ----------

# Keep only the columns we need for training model
pdf = pdf[['TICKET_ID', 'LONG_DESC_TEXT', 'ASSIGNED_OWNER_GROUP', 'REPORT_DATE', 'DEPARTMENT_SRC']]
pdf

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load new tickets (since last record in CSV) from SQL
# MAGIC We will join these into the original tickets. The original CSV has lots of columns used originally for EDA. We don't need those for actually training the model so we will remove them.

# COMMAND ----------

# last_date = '2020-01-01' # Ran this one-time to replace some bad values in the existing CSV. Keep this commented out from now on.

# Starting here, copied from 4_remove_pii notebook from original ecd-notebooks repo
df = spark.sql('''
    SELECT
        ticket.TICKET_ID,
        longdesc.LONG_DESC_TEXT,
        status.ASSIGNED_OWNER_GROUP,
        ticket.REPORT_DATE,
        ticket.DEPARTMENT_SRC
    FROM
        edr.DEMAND_ECD_TICKET_STATUS_HSTRY AS status
    INNER JOIN
        edr.DEMAND_ECD_TICKET as ticket ON status.ticket_id = ticket.ticket_id
    INNER JOIN
        edr.DEMAND_ECD_LONG_DESCRIPTION as longdesc ON ticket.TICKET_USER_ID = longdesc.LONG_DESC_KEY
    WHERE
        ticket_status_history_id IN (
            SELECT
                max(ticket_status_history_id)
            FROM
                edr.DEMAND_ECD_TICKET_STATUS_HSTRY
            WHERE
                assigned_owner_group <> 'ESI00011'
                and
                assigned_owner_group <> 'ESI00043'
            GROUP BY
                ticket_id
        )
        AND length(longdesc.LONG_DESC_KEY) > 2
        AND longdesc.LONG_DESC_OWNER_SRC_COL = 'DESCRIPTION'
        AND length(longdesc.LONG_DESC_TEXT) > 30
        AND length(status.assigned_owner_group) > 5
        AND ticket.STATUS in ('RESOLVED', 'CLOSED')
        AND ticket.REPORT_DATE >= "'''
    + last_date + '"')

df.cache()
print(df.count())

# COMMAND ----------

df = df.toPandas()
df.head()

# COMMAND ----------

from io import StringIO
from html.parser import HTMLParser
import re
import string

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    a = s.get_data()
    return a.replace('\n', '. ').replace('\r', '. ')

def clean_string(s, decontract=False, deaccent=False, depunctuate=False,
                lowercase=False, remove_stopwords=False, lemmatize=False, remove_numbers=False):
    result = s
    if lowercase:
        result = result.lower()
        
  # Bad character from SAS
    result = result.replace('¿', ' ')
  # Ensure that sentences are not concatenated together when originally separated by line breaks or <p>
    result = result.replace('<', ' <')
#     result = result.replace('</p><p>', ". ")
#     result = result.replace('</pre><pre>', ". ")
#     result = result.replace('<br>', ". ")
#     result = result.replace('<br />', ". ")
    result = strip_tags(result)
    result = result.replace(u'\ufffd', " ") # Switch unicode replacement character for space
    result = result.replace('\xa0', ' ') # A weird string common in the input data
    if decontract:
        result = result.replace("’","'")
        result = result.lower() # Tokenizer does this already, but this is necessary for decontraction
        result = decontracted(result)
    if depunctuate:
        translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) # Tokenizer does this already?
        result = result.translate(translator) # Tokenizer does this already?
    if deaccent:
        result = strip_accents(result)
    #   result = re.sub(r'\w*\d\w*', '', result).strip() # Remove all words containing numbers
    if remove_stopwords:
        tokens = word_tokenize(result)
        tokens = [t for t in tokens if t not in english_stopwords]
    if lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        result = ' '.join(tokens)
    if remove_numbers:
        result = re.sub(r"\b\d+\b", '', result).strip()
    return result

def clean_special_chars(text):
    # '@' symbol is removed by translation
#     text = text.replace('@', ' ATSYMBOL ')
    # Double spaces sometimes are used to break form fields
    text = text.replace('  ', '. ')
    # Remove the long strings of punctuation (e.g. ========)
    text = re.sub(r'[(\*?)+<>_=-]{2,}', '', text)
    # For slashes, remove when at least 3 (so http:// etc. is kept)
    text = re.sub(r'[\\/]{3,}', '', text)
    # Remove the long strings of spaces and periods (e.g. .  . .  .. )
    text = re.sub(r'[ .]{3,}', '. ', text)
    return text

# COMMAND ----------

descs = [clean_special_chars(clean_string(s)) for s in list(df['LONG_DESC_TEXT'])]

# Some manual filtering
good_idx = [i for i, s in enumerate(descs) if len(s) > 100 or (len(s) >= 25 and len(s) <= 100) and not (('reporter' in s.lower() or 'affected' in s.lower() or 'specifications' in s.lower() or 'name' in s.lower() or 'being notified of' in s.lower() or 'select the requested' in s.lower() or 'details' in s.lower() or 'info copied from email' in s.lower() or 'form' in s.lower() or 'attached' in s.lower() or 'summary' in s.lower()) and ('CI ' not in s))]

good_idx_set = set(good_idx)
bad_idx = [i for i in list(df.index) if i not in good_idx_set]
print(f'{len(bad_idx)*100 / len(df):.2f}% long descriptions are "bad"\n')

# print(list(df.iloc[bad_idx].sample(10)['LONG_DESC_TEXT']))

# Before trimming the dataframe to only "good" rows, we need to add our clean texts into the dataframe
df['LONG_DESC_CLEAN'] = descs

# Trim DF to only include tickets with a "good" long description
print(len(df), 'before removing "bad" long desc tickets')
df = df.iloc[good_idx]
print(len(df), 'after removing')

# COMMAND ----------

df['LONG_DESC_TEXT'] = df['LONG_DESC_CLEAN']
df['LONG_DESC_TEXT'].sample(10, replace=True)

# COMMAND ----------

# Starting here, copied from 4b_remove_pii_presidio notebook in original ecd-notebooks repo
from presidio_analyzer import AnalyzerEngine

# Set up the engine, loads the NLP module (spaCy model by default) and other PII recognizers
analyzer = AnalyzerEngine()

from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import RecognizerResult, OperatorConfig

# Initialize the engine:
engine = AnonymizerEngine()

# Intentionally leaving in Locations and URLs (that aren't IP addresses)
# Location information is, AFAIK, always related to a GC site, and is relevant to classification
entities = [
    'DATE_TIME',
    'EMAIL_ADDRESS',
    'IP_ADDRESS',
    'PERSON',
    'PHONE_NUMBER'
]

operators = {
    entity: OperatorConfig("replace", {"new_value": "***"})
    for entity in entities
}

def presidio_clean(text, verbose=False):
    # Note the language is English. It would be better to run this after translation.
    results = analyzer.analyze(text=text, entities=entities, language='en')
    result = engine.anonymize(text=text, analyzer_results=results, operators=operators)
    if verbose:
        print(results)
        print(result)
    return result.text

# COMMAND ----------

import timeit

clean_texts = []
start_time = timeit.default_timer()
for i, t in enumerate(df['LONG_DESC_TEXT'].values):
    clean_texts.append(presidio_clean(t))
    if i%500 == 0:
        percent = 100*(i+1)/len(df)
        t = timeit.default_timer() - start_time
        rate = (i+1)/t
        est_time_total = len(df)/rate
        est_time_remaining = est_time_total - t
        print(f'Processed {i+1}/{len(df)} texts ({percent:.2f}%) in {t:.2f}s. Est time remaining: {est_time_remaining/60:.2f}m')

# COMMAND ----------

df['LONG_DESC_TEXT'] = clean_texts
df = df[['TICKET_ID', 'LONG_DESC_TEXT', 'ASSIGNED_OWNER_GROUP', 'REPORT_DATE', 'DEPARTMENT_SRC']]

# COMMAND ----------

joined_df = pd.concat([df, pdf])
joined_df.drop_duplicates(subset=['TICKET_ID'], inplace=True)
pdf = joined_df
print(len(pdf), 'records in joined dataframe after dropping duplicates')
pdf['REPORT_DATE'] = pd.to_datetime(pdf['REPORT_DATE'])
print(pdf['REPORT_DATE'].min(), pdf['REPORT_DATE'].max())

pdf['DEPARTMENT_SRC'] = pdf['DEPARTMENT_SRC'].fillna('NoDept')

# COMMAND ----------

# MAGIC %md
# MAGIC **New step: Remove tickets which are now in "DO NOT USE" groups**

# COMMAND ----------

group_descriptions = pd.read_csv('/dbfs/FileStore/ecd/20220706___ECD_Groups.csv')
dead_groups = group_descriptions[group_descriptions['Description'].fillna('').str.lower().str.contains('deletion')]['Person Group'].values
print(dead_groups)

pdf = pdf[~(pdf['ASSIGNED_OWNER_GROUP'].isin(dead_groups))]
print(len(pdf), 'after dropping tickets in dead groups')

# COMMAND ----------

# Not sure why this needs to be saved then loaded a second later, but whatever!
pdf.to_parquet(f'/dbfs/FileStore/ecd/{base_file_name}.parquet')

# Update the CSV so that we only bring in new records from the SQL query next time this is run
pdf.to_csv(f"/dbfs/FileStore/ecd/{base_file_name}.csv")

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

# This file is what the next step (preprocessing factory) uses
df = spark.read.parquet(f"dbfs:/FileStore/ecd/{base_file_name}_deduped.parquet")
print(f'Total records: [{df.count()}]')


# COMMAND ----------


