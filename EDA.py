# Databricks notebook source
# MAGIC %md
# MAGIC # EDA Work
# MAGIC 
# MAGIC In this notebook that aim is to understand the data, weaknesses and strengths. 
# MAGIC There are a number of issues identified:
# MAGIC * The lack of long-text data for most of 2020 and 2021.
# MAGIC * There is a significant drop in the number of tickets for past year
# MAGIC * The rule for the changes in the OGs is not apparent
# MAGIC * The cutt-off date needs to be defined

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and functions

# COMMAND ----------

# MAGIC %sh sudo apt-get install -y graphviz

# COMMAND ----------

# %pip install tensorflow -q
%pip install gensim -q
%pip install azureml-sdk -q
%pip install pydot -q
%pip install graphviz -q
%pip install nltk -q

# COMMAND ----------

import os
import re
import gensim
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import string
import unicodedata
import time
import sys
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import gensim.downloader as api
from sklearn.feature_extraction.text import TfidfVectorizer
import collections
import pickle
import gzip
from os.path import exists
import pandas as pd
import gc
import time
from datetime import date
import azureml.core
from azureml.core import Workspace, Experiment, Run
from azureml.core.model import Model as AzureModel
from azureml.core.datastore import Datastore
from azureml.core.authentication import ServicePrincipalAuthentication
import datetime
import os
import tempfile
import datetime
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
english_stopwords = stopwords.words('english')

kv_name = "ScScCSV-DSAI-Lab-dev-kv"

def getCorpus():
#   corpus = api.load('text8')
#   model = Word2Vec(corpus)
#   return corpus, model
    model = api.load('glove-wiki-gigaword-100') # Best performing
#     model = api.load('fasttext-wiki-news-subwords-300')
    return None, model

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')

def cleanString(s, decontract=False, deaccent=False, depunctuate=False,
                lowercase=False, remove_stopwords=False, lemmatize=False, remove_numbers=False):
    result = s
    result = result.replace(u'\ufffd', " ") # Switch unicode replacement character for space
    result = result.replace('\xa0', ' ') # A weird string common in the input data
    result = re.sub('((http|https)\:\/\/)[\*a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', 'URLREMOVED', result) # Removing URL
    
    if lowercase:
        result = result.lower()
        
    result = result.replace('this incident needs to be assigned to  .', '')

    for og in li_OGs:
        result = result.replace(og.lower(), '')
    
    if decontract:
        result = result.replace("’","'")
        result = result.lower() # Tokenizer does this already, but this is necessary for decontraction
        result = decontracted(result)
    if depunctuate:
        cst_punctuation = string.punctuation.replace('!', '').replace('.', '').replace('?', '').replace('&', '')
        translator = str.maketrans(cst_punctuation, ' '*len(cst_punctuation)) # Tokenizer does this already?
        result = result.translate(translator) # Tokenizer does this already?
        result = result.replace("&", " and ")
        result = result.replace("  ", " ")
    if deaccent:
        result = strip_accents(result)
    if remove_stopwords:
        tokens = word_tokenize(result)
        tokens = [t for t in tokens if t not in english_stopwords]
    if lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        result = ' '.join(tokens)
    if remove_numbers:
        result = re.sub(r"\b\d+\b", '', result).strip()
    
    return result

# def plot(history):
#     acc = history.history['categorical_accuracy']
#     val_acc = history.history['val_categorical_accuracy']
# 
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
# 
#     epochs = range(1, len(acc) + 1)
# 
#     plt.plot(epochs, acc, 'bo', label='Training acc')
#     plt.plot(epochs, val_acc, 'b', label='Validation acc')
#     #   for i,j in zip(epochs, val_acc):
#     #     plt.annotate("%.4f" % j,xy=(i,j))
#     plt.legend()
# 
#     plt.figure()
#     plt.show()

# Load a classifier from disk
def load(path):
    if path is None:
        print("No path")
        return

    if not exists(path):
        print("Path no exists")
        return

    try:
        with gzip.open(path, 'rb') as f:
            content = f.read()
            classifier = pickle.loads(content)
            
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
                fd.write(classifier['model'])
                fd.flush()

                unserialized = models.load_model(fd.name)
            
            classifier['model'] = unserialized
    except Exception as e:
        print(repr(e))
        return 

    return classifier

# COMMAND ----------

MODELPATH = '/dbfs/mnt/ecd/models/'
if not os.path.exists(MODELPATH):
    os.makedirs(MODELPATH)


# COMMAND ----------

# pdf = pd.read_csv("/dbfs/FileStore/ecd/ecd_tickets_cleaned_2_more.csv")

# COMMAND ----------

# pdf['INTERNAL_PRIORITY'] = pdf['INTERNAL_PRIORITY'].astype(str)
# pdf.to_parquet('/dbfs/FileStore/ecd/ecd_tickets_cleaned_2_more.parquet')

# COMMAND ----------

# df = spark.read.parquet("dbfs:/FileStore/ecd/ecd_tickets_cleaned_2_more.parquet")
# 
# df = df.withColumn("REPORT_DATE_DT", F.to_timestamp(F.col('REPORT_DATE'), 'yyyy-MM-dd HH:mm:ss'))
# df = df.withColumn("REPORT_MONTH", F.regexp_replace(F.substring('REPORT_DATE', 0, 7), '-', '').cast(IntegerType()))
# display(df)

# COMMAND ----------

# df.registerTempTable("df_temp")
# df_deduped = sqlContext.sql('''
#     SELECT 
#         *
#     FROM (
#         SELECT 
#             *,
#             ROW_NUMBER() OVER (PARTITION BY ticket_id ORDER BY REPORT_DATE_DT DESC) AS rank
#         FROM df_temp
#     ) vo WHERE rank = 1
# ''')
# df_deduped.drop('rank').write.format('delta').mode('overwrite').option("overwriteSchema", "true").save('dbfs:/FileStore/ecd/ecd_tickets_cleaned_2.delta')

# COMMAND ----------

df = spark.read.format('delta').load("dbfs:/FileStore/ecd/ecd_tickets_cleaned_2.delta")
df.count()

# COMMAND ----------

display(df)

# COMMAND ----------

# import sparknlp
# from sparknlp.base import *
# from sparknlp.annotator import *
# from pyspark.ml import Pipeline

# documentAssembler = DocumentAssembler() \
#     .setInputCol("LONG_DESC_TEXT") \
#     .setOutputCol("document")

# cleanUpPatterns = ["<[^>]>"]

# documentNormalizer = DocumentNormalizer() \
#     .setInputCols("document") \
#     .setOutputCol("normalizedDocument") \
#     .setAction("clean") \
#     .setPatterns(cleanUpPatterns) \
#     .setReplacement(" ") \
#     .setPolicy("pretty_all") \
#     .setLowercase(True)

# pipeline = Pipeline().setStages([
#     documentAssembler,
#     documentNormalizer
# ])

# pipelineModel = pipeline.fit(df)

# result = pipelineModel.transform(df)

# COMMAND ----------

# display(result.filter('REPORT_MONTH = 202205').select('LONG_DESC_TEXT', "normalizedDocument.result"))

# COMMAND ----------

# result = result.withColumn('normalized_text', F.col("normalizedDocument.result")[0])

# COMMAND ----------

# display(result.select("normalized_text"))

# COMMAND ----------

# documentAssembler = DocumentAssembler() \
#     .setInputCol("normalized_text") \
#     .setOutputCol("document")

# sentenceDetector = SentenceDetectorDLApproach() \
#     .setInputCols(["document"]) \
#     .setOutputCol("sentences") \
#     .setEpochsNumber(100)

# pipeline = Pipeline().setStages([documentAssembler, sentenceDetector])
# model = pipeline.fit(result)


# COMMAND ----------

li_OGs = [x['ASSIGNED_OWNER_GROUP'].lower() for x in df.select('ASSIGNED_OWNER_GROUP').distinct().collect()]
li_OGs[0:10]

# COMMAND ----------

# cleanString(s, decontract=False, deaccent=False, depunctuate=True, lowercase=True, remove_stopwords=False, lemmatize=False, remove_numbers=False)
#                decontract=False, deaccent=False, depunctuate=False, lowercase=False, remove_stopwords=False, lemmatize=False, remove_numbers=False

udf_cleanString = udf(cleanString, StringType())
df = df.withColumn('LONG_DESC_TEXT_CLEANED', udf_cleanString(F.col('LONG_DESC_TEXT'), 
                                                             F.lit(False), 
                                                             F.lit(False), 
                                                             F.lit(True), 
                                                             F.lit(True), 
                                                             F.lit(False), 
                                                             F.lit(False), 
                                                             F.lit(False)))

# COMMAND ----------

df = df.withColumn('TEXT_FINAL', F.concat(F.col('DEPARTMENT_SRC'), F.lit(' '), F.col('LONG_DESC_TEXT_CLEANED')))
display(df.select('DEPARTMENT_SRC', 'LONG_DESC_TEXT_CLEANED', 'TEXT_FINAL'))


# COMMAND ----------

df.filter('TICKET_CLASS = "INCIDENT" and TICKET_ID = "IN10028698"').select("TICKET_ID", "LONG_DESC_TEXT", "LONG_DESC_TEXT_CLEANED").collect()


# COMMAND ----------

df.filter('REPORT_MONTH = 202205').filter('TICKET_CLASS = "SR"').select("ASSIGNED_OWNER_GROUP", "LONG_DESC_TEXT").limit(20).collect()


# COMMAND ----------

def does_contain_any_og(long_text):
    lower_long_text = long_text.lower()
    li_val_contains = [x for x in li_OGs if x.lower() in lower_long_text]
    if len(li_val_contains) > 0:
        return li_val_contains[0]
    
    return None

udf_does_contain_any_og = udf(does_contain_any_og, StringType())


# COMMAND ----------

df = df.withColumn("TEXT_IN_OG", udf_does_contain_any_og(F.col('LONG_DESC_TEXT')))
df = df.withColumn("TEXT_CONTAINS_ANY_OG", F.when(F.col('TEXT_IN_OG').isNull(), F.lit(False)).otherwise(F.lit(True)))

# COMMAND ----------

df = df.withColumn("TEXT_CONTAINS_OG", F.when(F.lower(F.col("LONG_DESC_TEXT")).contains(F.lower(F.col("ASSIGNED_OWNER_GROUP"))), F.lit(True)).otherwise(F.lit(False)))

df = df.filter('TICKET_CLASS != "PROBLEM"')
df.count()

# COMMAND ----------

display(df.select("LONG_DESC_TEXT", "LONG_DESC_TEXT_CLEANED", "TEXT_CONTAINS_ANY_OG", "TEXT_IN_OG", "ASSIGNED_OWNER_GROUP").filter('TEXT_CONTAINS_ANY_OG and not TEXT_CONTAINS_OG'))


# COMMAND ----------

df.filter("TEXT_CONTAINS_OG").count()

# COMMAND ----------

df.filter("TEXT_CONTAINS_ANY_OG").count()

# COMMAND ----------

display(df.groupby("REPORT_MONTH", "TEXT_CONTAINS_OG").agg(F.count('*').alias('cnt')).sort(F.col('REPORT_MONTH')))


# COMMAND ----------

display(df.groupby("REPORT_MONTH", "TEXT_CONTAINS_ANY_OG").agg(F.count('*').alias('cnt')).sort(F.col('REPORT_MONTH')))


# COMMAND ----------

display(df.groupby("REPORT_MONTH", "TEXT_CONTAINS_OG").agg(F.count('*').alias('cnt')).sort(F.col('REPORT_MONTH')))


# COMMAND ----------

display(df.groupby("REPORT_MONTH", "TEXT_CONTAINS_ANY_OG").agg(F.count('*').alias('cnt')).sort(F.col('REPORT_MONTH')))


# COMMAND ----------

display(df.groupby("REPORT_MONTH", "TEXT_CONTAINS_OG", 'TICKET_CLASS').agg(F.count('*').alias('cnt')).sort(F.col('REPORT_MONTH')))

# COMMAND ----------

display(df.groupby("REPORT_MONTH", "TEXT_CONTAINS_ANY_OG", 'TICKET_CLASS').agg(F.count('*').alias('cnt')).sort(F.col('REPORT_MONTH')))

# COMMAND ----------

display(df.groupby("REPORT_MONTH", 'TICKET_CLASS').agg(F.count('*').alias('cnt')).sort(F.col('REPORT_MONTH')))

# COMMAND ----------

# MAGIC %md
# MAGIC # 

# COMMAND ----------

# MAGIC %md
# MAGIC ## The total number of classes from the April 2022.

# COMMAND ----------

total_classes_202204 = df.filter(f'REPORT_MONTH == 202204').select("ASSIGNED_OWNER_GROUP").distinct().count()
print(f'Total classes in April 2022: {total_classes_202204}')

# COMMAND ----------

display(df.filter(f'REPORT_MONTH == 202204').groupby("ASSIGNED_OWNER_GROUP").agg(F.count('*').alias('cnt')).sort(F.desc('cnt')))

# COMMAND ----------

total_classes_202205 = df.filter(f'REPORT_MONTH == 202205').select("ASSIGNED_OWNER_GROUP").distinct().count()
print(f'Total classes in April 2022: {total_classes_202205}')

# COMMAND ----------

display(df.filter(f'REPORT_MONTH == 202205').groupby("ASSIGNED_OWNER_GROUP").agg(F.count('*').alias('cnt')).sort(F.desc('cnt')))

# COMMAND ----------

total_classes_from_202204 = df.filter(f'REPORT_MONTH >= 202204').select("ASSIGNED_OWNER_GROUP").distinct().count()
print(f'Total classes from April 2022: {total_classes_from_202204}')

# COMMAND ----------

display(df.filter(f'REPORT_MONTH >= 202204').groupby("ASSIGNED_OWNER_GROUP").agg(F.count('*').alias('cnt')).sort(F.desc('cnt')))

# COMMAND ----------

total_classes_between_202104_202204 = df.filter(f'REPORT_MONTH < 202204 and REPORT_MONTH >= 202104').select("ASSIGNED_OWNER_GROUP").distinct().count()
print(f'Total classes from April 2022: {total_classes_between_202104_202204}')

# COMMAND ----------

display(df.filter(f'REPORT_MONTH < 202204 and REPORT_MONTH >= 202104').groupby("ASSIGNED_OWNER_GROUP").agg(F.count('*').alias('cnt')).sort(F.desc('cnt')))

# COMMAND ----------

pdf_202104_202203 = df.filter(f'REPORT_MONTH < 202204 and REPORT_MONTH >= 202104').select("ASSIGNED_OWNER_GROUP").distinct().toPandas()
set_202104_202203 = set(pdf_202104_202203['ASSIGNED_OWNER_GROUP'])
set_202104_202203

# COMMAND ----------

pdf_202204 = df.filter(f'REPORT_MONTH >= 202204').select("ASSIGNED_OWNER_GROUP").distinct().toPandas()
set_202204 = set(pdf_202204['ASSIGNED_OWNER_GROUP'])
set_202204

# COMMAND ----------

print(len(set_202104_202203))
print(len(set_202204))

# COMMAND ----------

li_newly_202204 = list(set_202204.difference(set_202104_202203))
len(li_newly_202204)

# COMMAND ----------

display(df.filter(f'REPORT_MONTH >= 202204').filter(F.col('ASSIGNED_OWNER_GROUP').isin(li_newly_202204)).groupby("ASSIGNED_OWNER_GROUP").agg(F.count('*').alias('cnt')).sort(F.desc('cnt')))

# COMMAND ----------

df.filter(f'REPORT_MONTH >= 202204').filter(F.col('ASSIGNED_OWNER_GROUP').isin(li_newly_202204)).select("LONG_DESC_TEXT", "LONG_DESC_TEXT_CLEANED", "ASSIGNED_OWNER_GROUP", "TICKET_CLASS").sort(F.col('ASSIGNED_OWNER_GROUP')).collect()

# COMMAND ----------

pdf_before_202104 = df.filter(f'REPORT_MONTH < 202104').select("ASSIGNED_OWNER_GROUP").distinct().toPandas()
set_before_202104 = set(pdf_before_202104['ASSIGNED_OWNER_GROUP'])
set_before_202104

# COMMAND ----------

total_classes_202104 = df.filter(f'REPORT_MONTH >= 202104').select("ASSIGNED_OWNER_GROUP").distinct().count()
print(f'Total classes from April 2021: {total_classes_202104}')

# COMMAND ----------



# COMMAND ----------

display(df.filter(f'REPORT_MONTH >= 202104').groupby("ASSIGNED_OWNER_GROUP").agg(F.count('*').alias('cnt')).sort(F.desc('cnt')))

# COMMAND ----------

cut_off_date = 202107
display(df.filter(f'REPORT_MONTH >= {cut_off_date}').groupby("TICKET_CLASS").agg(F.count('*').alias('cnt')))

# COMMAND ----------

df.filter(f'REPORT_MONTH >= {cut_off_date}').count()

# COMMAND ----------

df.filter(f'REPORT_MONTH >= {cut_off_date}').select("ASSIGNED_OWNER_GROUP").distinct().count()

# COMMAND ----------

df.select("ASSIGNED_OWNER_GROUP").distinct().count()

# COMMAND ----------

cut_off_date_recent = 202107
cut_off_date_all_time = 201807

top_classes_recent = df.filter(f"REPORT_MONTH >= {cut_off_date_recent}").groupby('ASSIGNED_OWNER_GROUP').agg(F.count('*').alias('cnt')).sort(F.desc('cnt')).toPandas()
top_classes_all_time = df.filter(f"REPORT_MONTH >= {cut_off_date_all_time}").groupby('ASSIGNED_OWNER_GROUP').agg(F.count('*').alias('cnt')).sort(F.desc('cnt')).toPandas()

# COMMAND ----------

len(top_classes_recent), cut_off_date_recent

# COMMAND ----------

len(top_classes_all_time), cut_off_date_all_time

# COMMAND ----------

top_n = 20
classes_of_interest_all_time = list(top_classes_all_time['ASSIGNED_OWNER_GROUP'])[0:top_n]

df = df.withColumn('TOP_OG_ALL_TIME', F.when(F.col('ASSIGNED_OWNER_GROUP').isin(classes_of_interest_all_time), F.col('ASSIGNED_OWNER_GROUP')).otherwise(F.lit('OTHER')))
df = df.withColumn('TOP_OR_NOT_OG_ALL_TIME', F.when(F.col('ASSIGNED_OWNER_GROUP').isin(classes_of_interest_all_time), F.lit(f'TOP_{top_n}')).otherwise(F.lit('OTHER')))

# COMMAND ----------

classes_of_interest_recent = list(top_classes_recent['ASSIGNED_OWNER_GROUP'])[0:top_n]

df = df.withColumn('TOP_OG_RECENT', F.when(F.col('ASSIGNED_OWNER_GROUP').isin(classes_of_interest_recent), F.col('ASSIGNED_OWNER_GROUP')).otherwise(F.lit('OTHER')))
df = df.withColumn('TOP_OR_NOT_OG_RECENT', F.when(F.col('ASSIGNED_OWNER_GROUP').isin(classes_of_interest_recent), F.lit(f'TOP_{top_n}')).otherwise(F.lit('OTHER')))

# COMMAND ----------

display(df.groupby('REPORT_MONTH', "TOP_OG_ALL_TIME", "TICKET_CLASS").agg(F.count('*').alias('cnt')).sort(F.col('REPORT_MONTH')))


# COMMAND ----------

display(df.groupby('REPORT_MONTH', "TOP_OG_RECENT", "TICKET_CLASS").agg(F.count('*').alias('cnt')).sort(F.col('REPORT_MONTH')))


# COMMAND ----------

top_n = 50
classes_of_interest_recent = list(top_classes_recent['ASSIGNED_OWNER_GROUP'])[0:top_n]

df = df.withColumn('TOP_OG_RECENT', F.when(F.col('ASSIGNED_OWNER_GROUP').isin(classes_of_interest_recent), F.col('ASSIGNED_OWNER_GROUP')).otherwise(F.lit('OTHER')))
df = df.withColumn('TOP_OG_ALL_TIME', F.when(F.col('ASSIGNED_OWNER_GROUP').isin(classes_of_interest_recent), F.col('ASSIGNED_OWNER_GROUP')).otherwise(F.lit('OTHER')))

# COMMAND ----------

cut_off_for_training = 201808

# COMMAND ----------

display(df.filter(f"REPORT_MONTH >= {cut_off_for_training}").groupby('TOP_OG_RECENT').agg(F.count('*').alias('cnt')).sort(F.desc('cnt')))

# COMMAND ----------

df_target = df.filter(f"REPORT_MONTH >= {cut_off_for_training}").select('TEXT_FINAL', 'TOP_OG_RECENT')
df_target.count()

# COMMAND ----------

display(df_target)


# COMMAND ----------

pdf_target = df_target.toPandas()

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_test_val, y_train, y_test_val = train_test_split(pdf_target['TEXT_FINAL'], pdf_target['TOP_OG_RECENT'],
                                                    stratify=pdf_target['TOP_OG_RECENT'],
                                                    shuffle=True,
                                                    test_size=0.20)

# COMMAND ----------

X_val, X_test, y_val, y_test = train_test_split(X_test_val.to_frame()['TEXT_FINAL'], y_test_val,
                                                    stratify=y_test_val,
                                                    shuffle=True,
                                                    test_size=0.5)

# COMMAND ----------

X_train.shape, X_val.shape, X_test.shape

# COMMAND ----------

pdf_X_train = X_train.to_frame()
pdf_X_val = X_val.to_frame()
pdf_X_test = X_test.to_frame()

pdf_X_train['target'] = y_train
pdf_X_val['target'] = y_val
pdf_X_test['target'] = y_test

# COMMAND ----------

display(pdf_X_train['target'].value_counts().to_frame())

# COMMAND ----------

display(pdf_X_val['target'].value_counts().to_frame())

# COMMAND ----------

display(pdf_X_test['target'].value_counts().to_frame())

# COMMAND ----------

import os
import pandas as pd
import azureml.core
import numpy as np
from azureml.core import Workspace, Environment, Experiment, Datastore, Dataset, ScriptRunConfig
from azureml.train.automl.run import AutoMLRun
from azureml.core.datastore import Datastore
from azureml.data.data_reference import DataReference

# Check core SDK version number
print("SDK version:", azureml.core.VERSION)


# COMMAND ----------

from azureml.core.authentication import InteractiveLoginAuthentication

interactive_auth = InteractiveLoginAuthentication(tenant_id="8c1a4d93-d828-4d0e-9303-fd3bd611c822")
ws = Workspace(subscription_id="105efa68-0ff4-486f-ae3a-86e28a447237",
               resource_group="ScSc-DSAI-AIDE-dev-rg",
               workspace_name="ScScCPS-DSAI-AIDE-dev-mlw", 
               auth=interactive_auth)

# COMMAND ----------

def_blob_store = ws.get_default_datastore()

# COMMAND ----------

top_n, cut_off_for_training
ds_X_train = Dataset.Tabular.register_pandas_dataframe(dataframe=pdf_X_train, target=(def_blob_store, 'nlp_automl'), name="owner_g_classfication_train", tags = {'top': top_n, 'period': f'REPORT_MONTH >= {cut_off_for_training}', 'ratio': '80%'})
ds_X_val = Dataset.Tabular.register_pandas_dataframe(dataframe=pdf_X_val, target=(def_blob_store, 'nlp_automl'), name="owner_g_classfication_val", tags = {'top': top_n, 'period': f'REPORT_MONTH >= {cut_off_for_training}', 'ratio': '10%'})
ds_X_test = Dataset.Tabular.register_pandas_dataframe(dataframe=pdf_X_test, target=(def_blob_store, 'nlp_automl'), name="owner_g_classfication_test", tags = {'top': top_n, 'period': f'REPORT_MONTH >= {cut_off_for_training}', 'ratio': '10%'})


# COMMAND ----------

pdf_X_train.to_csv('/dbfs/FileStore/ecd/pdf_X_train.csv', encoding='utf-8', index=False)
pdf_X_val.to_csv('/dbfs/FileStore/ecd/pdf_X_val.csv', encoding='utf-8', index=False)
pdf_X_test.to_csv('/dbfs/FileStore/ecd/pdf_X_test.csv', encoding='utf-8', index=False)


# COMMAND ----------

df_month_aggregation = df.filter('REPORT_MONTH = 202205').groupby('ASSIGNED_OWNER_GROUP').agg(F.count('*').alias('cnt'))
display(df_month_aggregation.sort(F.desc('cnt')))

# COMMAND ----------

print(df.filter('REPORT_MONTH = 202205').select('ASSIGNED_OWNER_GROUP').distinct().count())
print(df.filter('REPORT_MONTH > 202201').select('ASSIGNED_OWNER_GROUP').distinct().count())

# COMMAND ----------

df.select('ASSIGNED_OWNER_GROUP').distinct().count()

# COMMAND ----------

df_month_aggregation = df.groupby('ASSIGNED_OWNER_GROUP').agg(F.count('*').alias('cnt'))
display(df_month_aggregation.sort(F.desc('cnt')))

# COMMAND ----------

df_month_aggregation = df.groupby('REPORT_MONTH').agg(F.count('*').alias('cnt')).filter('REPORT_MONTH >= 201604')
pdf_month_aggregation = df_month_aggregation.sort(F.col("REPORT_MONTH")).toPandas()

# COMMAND ----------



# COMMAND ----------

trace_1 = go.Scatter(
    x=pdf_month_aggregation.REPORT_MONTH,
    y=pdf_month_aggregation.cnt,
    name = "Counts of Tickets",
    line = dict(color = '#FFA833'),
    opacity = 0.9)

data = [trace_1]

layout = dict(
    title='Counts of Tickets Over Time',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1y',
                     step='year',
                     stepmode='backward'),
                dict(count=5,
                     label='5y',
                     step='year',
                     stepmode='backward'),
                dict(count=10,
                     label='10y',
                     step='year',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(
            visible = True
        ),
        type='date'
    )
)

# fig = dict(data=data, layout=layout)
# displayHTML(plot(fig, output_type='div'))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

top_classes = df.filter("REPORT_MONTH = 202205").groupby('ASSIGNED_OWNER_GROUP').agg(F.count('*').alias('cnt')).sort(F.desc('cnt')).toPandas()
top_n = 50
classes_of_interest = list(top_classes['ASSIGNED_OWNER_GROUP'])[0:top_n]

df = df.withColumn('TOP_OG', F.when(F.col('ASSIGNED_OWNER_GROUP').isin(classes_of_interest), F.col('ASSIGNED_OWNER_GROUP')).otherwise(F.lit('OTHER')))
df = df.withColumn('TOP_OR_NOT_OG', F.when(F.col('ASSIGNED_OWNER_GROUP').isin(classes_of_interest), F.lit(f'TOP_{top_n}')).otherwise(F.lit('OTHER')))

# COMMAND ----------

df.filter('TOP_OG = "OTHER"').count()

# COMMAND ----------

df_month_aggregation = df.filter('REPORT_MONTH > 201808').groupby('REPORT_MONTH', "TOP_OG", "TICKET_CLASS").agg(F.count('*').alias('cnt')).filter('REPORT_MONTH >= 201604')

display(df_month_aggregation)

# COMMAND ----------

display(df_month_aggregation)

# COMMAND ----------

df_month_aggregation = df.filter('REPORT_MONTH > 201808').groupby('REPORT_MONTH', "TOP_OR_NOT_OG").agg(F.count('*').alias('cnt')).filter('REPORT_MONTH >= 201604')
display(df_month_aggregation.sort(F.col('REPORT_MONTH')))


# COMMAND ----------

display(df_month_aggregation.sort(F.col('REPORT_MONTH')))


# COMMAND ----------

pdf_month_aggregation.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data distributions
# MAGIC Not shown here: French text is often mixed into the English long description text.

# COMMAND ----------

# Distribution of tickets by quarter
def qdist(all_tickets):
    all_tickets['quarter'] = all_tickets['REPORT_DATE'].dt.year.astype(str) + '_' + all_tickets['REPORT_DATE'].dt.quarter.astype(str)
    dfs3 = pd.DataFrame(all_tickets.quarter.value_counts()).reset_index()
    dfs3.columns = ['quarter', 'num records']
    return dfs3.sort_values(['quarter']).plot.bar(x='quarter')

# Note that the jagged distribution is an artifact of the join to Long Descriptions.
# In reality, the distribution of tickets over quarters is fairly uniform.
qdist(df)

# COMMAND ----------

# How long are the "Long descriptions"?
# Judging by these results we should set truncate at 300 words.
lens = [len(d.split()) for d in df['LONG_DESC_TEXT']]
pd.Series(lens).hist()

# COMMAND ----------

# Label distributions over time?
group_counts = dict(df.query('REPORT_DATE >= "2021-04-01"')['ASSIGNED_OWNER_GROUP'].value_counts()[:50])
df_only_labels = df[['REPORT_DATE', 'ASSIGNED_OWNER_GROUP']]
df_only_labels['quarter'] = df_only_labels['REPORT_DATE'].dt.year.astype(str) + '_' + df_only_labels['REPORT_DATE'].dt.quarter.astype(str)
df_only_labels.loc[~(df_only_labels['ASSIGNED_OWNER_GROUP'].isin(list(group_counts.keys()))), 'ASSIGNED_OWNER_GROUP'] = 'Other'

groups_over_quarters = df_only_labels.groupby(['ASSIGNED_OWNER_GROUP', 'quarter'])[['REPORT_DATE']].count().reset_index()
groups_over_quarters.columns = ['owner', 'quarter', 'count']

groups_over_quarters = groups_over_quarters.sort_values(['quarter', 'owner'])

# Note that ITS312 and ITS322 - common in 2016 - are completely absent in 2021.
display(groups_over_quarters)

# COMMAND ----------

# Print a few examples
# Spaces are irrelevant (removed in Tokenizer)
print(list(df.sample(5)['LONG_DESC_TEXT'].values))
