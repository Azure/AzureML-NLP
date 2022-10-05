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

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.context import SparkContext

import azureml.core
from azureml.core import Workspace, Run, Datastore, Dataset
from azureml.core.datastore import Datastore
from azureml.data.data_reference import DataReference

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


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

    parser.add_argument("--cut_off_for_training", type=int, default=201808, help="the cut off date for training dataset")
    parser.add_argument("--valid_classes_period", type=int, default=202105, help="start date as the cut off for valid classes, default is May 2021 as we start to get some data after a period.")
    parser.add_argument("--cut_off_date_recent", type=int, default=202107, help="start date to generate top N classes")
    parser.add_argument("--temporal_test_date", type=int, default=202206, help="value to filter the temporal test - this typically is the last month available for hyperparameter tuning exercise")
    parser.add_argument("--logic_v", type=int, default=2, help="a numerical value between 2 and 4 which corresponds to the logic behind the steps")
    parser.add_argument("--top_n", type=int, default=120, help="top n classes - for example, n=120 produces a dataset that contains 121 classes (120 top frequent owner groups and the rest as others)")
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
    
    li_OGs_i = brd_li_OGs.value
    for og in li_OGs_i:
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


def generate_target_df(df, cut_off_date_recent, top_n, temporal_test_date=202206, cut_off_for_training=201808, logic=2, valid_classes_period = 202105):
    pdf_top_classes_recent = df.filter(f"REPORT_MONTH >= {cut_off_date_recent} and REPORT_MONTH < {temporal_test_date}").groupby('ASSIGNED_OWNER_GROUP').agg(F.count('*').alias('cnt')).sort(F.desc('cnt')).toPandas()
    classes_of_interest_recent = list(pdf_top_classes_recent['ASSIGNED_OWNER_GROUP'])[0:top_n]

    df = df.withColumn('TOP_OG', F.when(F.col('ASSIGNED_OWNER_GROUP').isin(classes_of_interest_recent), F.col('ASSIGNED_OWNER_GROUP')).otherwise(F.lit('OTHER')))
        
    if logic == 3:
        cut_off_for_training = valid_classes_period
        
    df = df.filter(f"REPORT_MONTH >= {cut_off_for_training}")
    if logic == 4:
        df = df.filter(f'TOP_OG != "OTHER" OR (TOP_OG == "OTHER" and REPORT_MONTH >= {cut_off_date_recent})')
    
    return df, classes_of_interest_recent


def clean_base_df(df):
    global brd_li_OGs
    li_OGs = [x['ASSIGNED_OWNER_GROUP'].lower() for x in df.select('ASSIGNED_OWNER_GROUP').distinct().collect()]
    brd_li_OGs = sc.broadcast(li_OGs)
    print(f'Total Owner Groups: {len(li_OGs)}')
    
    udf_cleanString = udf(cleanString, StringType())
    df = df.withColumn('LONG_DESC_TEXT_CLEANED', udf_cleanString(F.col('LONG_DESC_TEXT'), 
                                                                 F.lit(False),
                                                                 F.lit(False),
                                                                 F.lit(True),
                                                                 F.lit(True),
                                                                 F.lit(False),
                                                                 F.lit(False),
                                                                 F.lit(False)))
    
    df = df.withColumn('TEXT_FINAL', F.concat(F.coalesce(F.col('DEPARTMENT_SRC'), F.lit('')), F.lit(' '), F.coalesce(F.col('LONG_DESC_TEXT_CLEANED'), F.lit(''))))
    # df = df.filter('TICKET_CLASS != "PROBLEM"')
    df = df.filter('TEXT_FINAL is not null and ASSIGNED_OWNER_GROUP is not null')
    return df, li_OGs


def prep_target_df(cut_off_for_training = 201808, 
                   valid_classes_period = 202105, 
                   cut_off_date_recent = 202107, 
                   temporal_test_date = 202206, 
                   logic_v = 2, 
                   top_n = 120):
    '''
    @param cut_off_for_training: the cut off date for training dataset
    @param valid_classes_period: for 202105 as the cut off for valid classes, May 2021 is when it starts to get some data after a period.
    @param cut_off_date_recent: Keeping 1 year of data, from July 21 to the end of June 22 as June 22 as test data to generate the top N classes
    @param temporal_test_date: value to filter the temporal test - this typically is the last month available for hyperparameter tuning exercise
    @param logic_v: a numerical value between 2 and 4 which corresponds to the logic behind the steps
    @param top_n: top n classes - for example, n=120 produces a dataset that contains 121 classes (120 top frequent owner groups and the rest as others)
    @return: three valuues, target training spark dataframe, target temporal spark dataframe, a list of values for department sources as new tokens
    '''
    
    set_valid_classes = set(df.filter(f"REPORT_MONTH >= {valid_classes_period}").select('ASSIGNED_OWNER_GROUP').distinct().toPandas()['ASSIGNED_OWNER_GROUP'])
    print(f'Total no. of valid OGs: [{len(set_valid_classes)}]')

    df_interest = df.filter(F.col('ASSIGNED_OWNER_GROUP').isin(list(set_valid_classes)))
    print(f'Total records with valid classes: [{df_interest.count()}]')

    # df, cut_off_date_recent, top_n, temporal_test_date=202206, cut_off_for_training=201808, logic=2, valid_classes_period = 202105
    df_interest, classes_of_interest_recent = generate_target_df(df_interest, cut_off_date_recent=cut_off_date_recent, top_n=top_n, temporal_test_date=temporal_test_date, cut_off_for_training=cut_off_for_training, logic=logic_v, valid_classes_period=valid_classes_period)
    # df_interest, classes_of_interest_recent = generate_target_df(df_interest, cut_off_date_recent, top_n, temporal_test_date, cut_off_for_training, logic_v, valid_classes_period)
    
    print(f'Total records for generated: [{df_interest.count()}]')

    df_target = df_interest.filter(f"REPORT_MONTH >= {cut_off_for_training}").select('TEXT_FINAL', 'TOP_OG', 'DEPARTMENT_SRC', 'REPORT_MONTH', 'ASSIGNED_OWNER_GROUP')
    print(f'Total records for targeted df: [{df_target.count()}]')

    li_dep_src = [x['DEPARTMENT_SRC'].lower() for x in df_target.filter('DEPARTMENT_SRC is not null').select('DEPARTMENT_SRC').distinct().collect()]
    print(f'Total department source as new tokens: [{len(li_dep_src)}]')

    df_temporal_test = df_target.filter(f'REPORT_MONTH == {temporal_test_date}')
    df_target_training = df_target.filter(f'REPORT_MONTH < {temporal_test_date}')

    print(f'Total records for temporal df: [{df_temporal_test.count()}]')
    print(f'Total records for target training df: [{df_target_training.count()}]')
    # df_interest.count(), df_target.count(), df_target_training.count(), df_temporal_test.count()
    
    return df_target, df_target_training, df_temporal_test, li_dep_src


def generate_top_ogs(df_target_training, df_temporal_test, run, label = 'TOP_OG'):
    label = 'TOP_OG'
    pdf_target_training = df_target_training.toPandas()

    X_train, X_test_val, y_train, y_test_val = train_test_split(pdf_target_training.drop(label, axis=1), pdf_target_training[label],
                                                        stratify=pdf_target_training[label],
                                                        shuffle=True,
                                                        test_size=0.20)

    X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val,
                                                        stratify=y_test_val,
                                                        shuffle=True,
                                                        test_size=0.5)
    pdf_X_train = X_train
    pdf_X_val = X_val
    pdf_X_test = X_test

    pdf_X_train['target'] = y_train
    pdf_X_val['target'] = y_val
    pdf_X_test['target'] = y_test
    pdf_temporal_test = df_temporal_test.withColumnRenamed(label, 'target').toPandas()
    
    print(f'Total records for: "pdf_X_train": [{pdf_X_train.shape[0]}]')
    print(f'Total records for: "pdf_X_val": [{pdf_X_val.shape[0]}]')
    print(f'Total records for: "pdf_X_test": [{pdf_X_test.shape[0]}]')
    print(f'Total records for: "pdf_temporal_test": [{pdf_temporal_test.shape[0]}]')

    run.log("pdf_X_train_size", pdf_X_train.shape[0])
    run.log("pdf_X_val_size", pdf_X_val.shape[0])
    run.log("pdf_X_test_size", pdf_X_test.shape[0])
    run.log("pdf_temporal_test_size", pdf_temporal_test.shape[0])
    
    return pdf_X_train, pdf_X_val, pdf_X_test, pdf_temporal_test


def generate_tags(top_n, pdf, cut_off_for_training, valid_classes_period, cut_off_date_recent, temporal_test_date, logic_v, li_OGs, li_dep_src):
    description = ''
    if logic_v == 2:
        description = f'records of the retired classes removed'
    elif logic_v == 3:
        description = f'records of the retired classes removed - training cut off is set to {cut_off_for_training}'
    elif logic_v == 4:
        description = f'all records since [{cut_off_for_training}] but OTHER from [{cut_off_date_recent}]'
        
    return {'top': top_n, 'total_records': pdf.shape[0], 'period': f'REPORT_MONTH >= {cut_off_for_training}', 'ratio': '80%', 'description': description, 'valid_period': valid_classes_period, 'logic': f'v{logic_v}', 'temporal_test_date' : temporal_test_date, 'owner groups': li_OGs, 'new_tokens': li_dep_src}


def register_aml_dataset(ws, pdf_X_train, pdf_X_val, pdf_X_test, pdf_temporal_test, tag_train, tag_val, tag_test, tag_temporal_test):
    '''
    Registeres the pandas dataframes, as AML Datasets and returning the dataset objects
    '''
    def_blob_store = ws.get_default_datastore()
    
    ds_X_train = Dataset.Tabular.register_pandas_dataframe(dataframe=pdf_X_train, target=(def_blob_store, 'nlp_automl'), name="owner_g_classfication_train", tags = tag_train)
    ds_X_val = Dataset.Tabular.register_pandas_dataframe(dataframe=pdf_X_val, target=(def_blob_store, 'nlp_automl'), name="owner_g_classfication_val", tags = tag_val)
    ds_X_test = Dataset.Tabular.register_pandas_dataframe(dataframe=pdf_X_test, target=(def_blob_store, 'nlp_automl'), name="owner_g_classfication_test", tags = tag_test)
    ds_temporal_test = Dataset.Tabular.register_pandas_dataframe(dataframe=pdf_temporal_test, target=(def_blob_store, 'nlp_automl'), name="owner_g_classfication_temporal_test", tags = tag_temporal_test)
    
    return ds_X_train, ds_X_val, ds_X_test, ds_temporal_test


def plot_og_trend(df, logic):
    df_og_trend = df.groupby('TOP_OG', 'REPORT_MONTH').agg(F.count('*').alias('cnt'))
    pdf_og_trend = df_og_trend.sort('REPORT_MONTH').toPandas()

    li_REPORT_MONTH = list(pdf_og_trend['REPORT_MONTH'].unique())
    li_OGs = list(pdf_og_trend['TOP_OG'].unique())

    pdf_REPORT_MONTH = pd.DataFrame(li_REPORT_MONTH)
    pdf_REPORT_MONTH.columns = ['REPORT_MONTH']

    dic_og_values = {}
    for og in li_OGs:
        pdf_og_target = pdf_og_trend[pdf_og_trend['TOP_OG'] == og]
        pdf_target_merged = pd.merge(pdf_REPORT_MONTH, pdf_og_target, on='REPORT_MONTH', how='left')[['REPORT_MONTH', 'cnt']].fillna(0)
        dic_og_values[og] = list(pdf_target_merged['cnt'])

    data = []
    for og in dic_og_values:
        trend = go.Scatter(
            x=[str(x) for x in li_REPORT_MONTH],
            y=dic_og_values[og],
            name = og,
            opacity = 0.9)
        data.append(trend)

    layout = dict(
        title=f'Rise and fall of OGs [Top {len(li_OGs)} OGs] - Logic [V{logic}]',
        width=2000,
        height=1000
    )

    fig = go.Figure(dict(data=data, layout=layout))
    return fig

def plot_og_dist(df_base, logic, second_cut_off_date = 0, temporal_test_date = 202206):
    df_plot_target = df_base.filter(f'REPORT_MONTH < {temporal_test_date}').filter(f'"{second_cut_off_date}" == 0 or REPORT_MONTH >= {second_cut_off_date}')

    total_records = df_plot_target.count()
    min_month = df_plot_target.select(F.min('REPORT_MONTH').alias('min_REPORT_MONTH')).collect()[0]['min_REPORT_MONTH']

    df_og_dist = df_plot_target.groupby('TOP_OG').agg(F.count('*').alias('cnt')).sort(F.desc('cnt'))
    df_og_dist = df_og_dist.withColumn('part_rate', F.col('cnt') / F.lit(total_records))

    pdf_og_dist = df_og_dist.toPandas()
    
    n = len(pdf_og_dist['TOP_OG'].unique())
    
    pdf_max = pdf_og_dist[pdf_og_dist.cnt == pdf_og_dist.cnt.max()]
    print(f"Most frequent OG: {pdf_max['TOP_OG'].iloc[0]} - [{pdf_max['cnt'].iloc[0]}]")
    
    pdf_min = pdf_og_dist[pdf_og_dist.cnt == pdf_og_dist.cnt.min()]
    print(f"Least frequent OG {pdf_min['TOP_OG'].iloc[0]} - [{pdf_min['cnt'].iloc[0]}]")
    
    fig_dist = dict(data=[go.Bar(x=pdf_og_dist['TOP_OG'], y=pdf_og_dist['cnt'])], layout=dict(title=f'Distribution of top [{n}] classes - Logic [V{logic}] - since [{min_month}] until [{temporal_test_date}]', width=2000, height=1000))
    fig_part_rate = dict(data=[go.Bar(x=pdf_og_dist['TOP_OG'], y=pdf_og_dist['part_rate'])], layout=dict(title=f'Participation rate of top [{n}] classes - Logic [V{logic}] - since [{min_month}] until [{temporal_test_date}]', width=2000, height=1000))
    
    return go.Figure(fig_dist), go.Figure(fig_part_rate)


def prepare_base_dataset(base_file_name):
    pdf = pd.read_csv(f"/dbfs/FileStore/ecd/{base_file_name}.csv")

    run.log('csv_file_size', pdf.shape[0])

    pdf['INTERNAL_PRIORITY'] = pdf['INTERNAL_PRIORITY'].astype(str)
    pdf['TICKET_SOURCE'] = pdf['TICKET_SOURCE'].astype(str)
    pdf['SELF_SERVICE_SOLUTION_FLAG'] = pdf['SELF_SERVICE_SOLUTION_FLAG'].astype(str)
    pdf['ACTUAL_COMPLETION_HRS'] = pdf['ACTUAL_COMPLETION_HRS'].astype(str)
    pdf['BUSINESS_COMPLETION_HRS'] = pdf['BUSINESS_COMPLETION_HRS'].astype(str)
    if 'CLASS_STRUCTURE_ID' in pdf.columns:
        pdf['CLASS_STRUCTURE_ID'] = pdf['CLASS_STRUCTURE_ID'].astype(str)

    # pdf.to_parquet('/dbfs/FileStore/ecd/ecd_tickets_cleaned_2_more.parquet')
    pdf.to_parquet(f'/dbfs/FileStore/ecd/{base_file_name}.parquet')

    df = spark.read.parquet(f"dbfs:/FileStore/ecd/{base_file_name}.parquet")

    df = df.withColumn("REPORT_DATE_DT", F.to_timestamp(F.col('REPORT_DATE'), 'yyyy-MM-dd HH:mm:ss'))
    df = df.withColumn("REPORT_MONTH", F.regexp_replace(F.substring('REPORT_DATE', 0, 7), '-', '').cast(IntegerType()))

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

    df = spark.read.parquet(f"dbfs:/FileStore/ecd/{base_file_name}_deduped.parquet")
    cnt_base = df.count()
    return df


if __name__ == '__main__':
    spark = SparkSession.builder.getOrCreate()
    sc = SparkContext.getOrCreate()
    args, extra_args = populate_environ()

    print(f'args: {args}')

    run = Run.get_context(allow_offline=False)
    print(f"Parent run id: [{run.parent.id}]")

    ws = run.experiment.workspace

    cut_off_for_training = args.cut_off_for_training
    valid_classes_period = args.valid_classes_period
    cut_off_date_recent = args.cut_off_date_recent
    temporal_test_date = args.temporal_test_date
    logic_v = args.logic_v
    top_n = args.top_n
    
    # ds_base = Dataset.get_by_name(ws, name=input_base_dataset_name, version=input_base_dataset_version)
    # pdf_base = ds_base.to_pandas_dataframe()

    # pdf_base.to_parquet('/dbfs/FileStore/ecd/temp.parquet')

    # df_base = prepare_base_dataset(args.base_file_name)
    base_file_name = args.base_file_name
    df_base = spark.read.parquet(f"dbfs:/FileStore/ecd/{base_file_name}_deduped.parquet")

    # df_base = spark.read.parquet('dbfs:/FileStore/ecd/temp.parquet')
    base_cnt = df_base.count()

    print(f'Size of the base dataframe: [{base_cnt}]')
    run.log('base_count', base_cnt)

    df, li_OGs = clean_base_df(df_base)
    df_cnt = df.count()
    
    print(f'Size of the cleaned dataframe: [{df_cnt}]')
    run.log('cleaned_df_count', df_cnt)

    df_target, df_target_training, df_temporal_test, li_dep_src = prep_target_df(cut_off_for_training, 
                                                                                 valid_classes_period, 
                                                                                 cut_off_date_recent, 
                                                                                 temporal_test_date, 
                                                                                 logic_v, 
                                                                                 top_n)
    

    fig_trend = plot_og_trend(df_target, logic_v)
    fig_dist, fig_part_rate = plot_og_dist(df_target, logic_v, second_cut_off_date = 0, temporal_test_date = temporal_test_date)

    if not os.path.exists("outputs/images"):
        os.makedirs("outputs/images")
        
    fig_trend.write_image("outputs/images/trend.jpeg")
    fig_dist.write_image("outputs/images/dist.jpeg")
    fig_part_rate.write_image("outputs/images/part_rate.jpeg")

    run.log_image('trend', path="outputs/images/trend.jpeg")
    run.log_image('dist', path="outputs/images/dist.jpeg")
    run.log_image('part_rate', path="outputs/images/part_rate.jpeg")

    pdf_X_train, pdf_X_val, pdf_X_test, pdf_temporal_test = generate_top_ogs(df_target_training, df_temporal_test, run, label = 'TOP_OG')
    
    print(f'Top n OGs: {top_n + 1}')
    assert(len(pdf_X_train['target'].unique()) == top_n + 1)

    # pdf_X_train.to_parquet(output_train)
    # pdf_X_val.to_parquet(output_validation)
    # pdf_X_test.to_parquet(output_test)
    # pdf_temporal_test.to_parquet(output_temporal_test)

    # df_X_train = spark.createDataFrame(pdf_X_train)
    # df_X_train.write.mode('overwrite').parquet(output_train)
    # df_X_val = spark.createDataFrame(pdf_X_val)
    # df_X_val.write.mode('overwrite').parquet(output_validation)
    # df_X_test = spark.createDataFrame(pdf_X_test)
    # df_X_test.write.mode('overwrite').parquet(output_test)
    # df_temporal_test = spark.createDataFrame(pdf_temporal_test)
    # df_temporal_test.write.mode('overwrite').parquet(output_temporal_test)
    
    tag_train = generate_tags(top_n, pdf_X_train, cut_off_for_training, valid_classes_period, cut_off_date_recent, temporal_test_date, logic_v, li_OGs, li_dep_src)
    tag_val = generate_tags(top_n, pdf_X_val, cut_off_for_training, valid_classes_period, cut_off_date_recent, temporal_test_date, logic_v, li_OGs, li_dep_src)
    tag_test = generate_tags(top_n, pdf_X_test, cut_off_for_training, valid_classes_period, cut_off_date_recent, temporal_test_date, logic_v, li_OGs, li_dep_src)
    tag_temporal_test = generate_tags(top_n, pdf_temporal_test, cut_off_for_training, valid_classes_period, cut_off_date_recent, temporal_test_date, logic_v, li_OGs, li_dep_src)
    
    ds_X_train, ds_X_val, ds_X_test, ds_temporal_test = register_aml_dataset(ws, pdf_X_train, pdf_X_val, pdf_X_test, pdf_temporal_test, tag_train, tag_val, tag_test, tag_temporal_test)
    
    print(f'{ds_X_train.tags}: V{ds_X_train.version}')
    print(f'{ds_X_val.tags}: V{ds_X_val.version}')
    print(f'{ds_X_test.tags}: V{ds_X_test.version}')
    print(f'{ds_temporal_test.tags}: V{ds_temporal_test.version}')
