# Databricks notebook source
import os
import pandas as pd
import azureml.core
import numpy as np
from azureml.core import Workspace, Environment, Experiment, Datastore, Dataset, ScriptRunConfig
from azureml.core.datastore import Datastore
from azureml.data.data_reference import DataReference

# Check core SDK version number
print("SDK version:", azureml.core.VERSION)


# COMMAND ----------

pdf_X_train = pd.read_csv('/dbfs/FileStore/ecd/pdf_X_train.csv')
pdf_X_val = pd.read_csv('/dbfs/FileStore/ecd/pdf_X_val.csv')
pdf_X_test = pd.read_csv('/dbfs/FileStore/ecd/pdf_X_test.csv')


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

ds_X_train = Dataset.Tabular.register_pandas_dataframe(dataframe=pdf_X_train, target=(def_blob_store, 'nlp_automl'), name="owner_g_classfication_train", tags = {'top': 100, 'period': 'REPORT_MONTH >= 202107', 'ratio': '80%'})
ds_X_val = Dataset.Tabular.register_pandas_dataframe(dataframe=pdf_X_val, target=(def_blob_store, 'nlp_automl'), name="owner_g_classfication_val", tags = {'top': 100, 'period': 'REPORT_MONTH >= 202107', 'ratio': '10%'})
ds_X_test = Dataset.Tabular.register_pandas_dataframe(dataframe=pdf_X_test, target=(def_blob_store, 'nlp_automl'), name="owner_g_classfication_test", tags = {'top': 100, 'period': 'REPORT_MONTH >= 202107', 'ratio': '10%'})

# COMMAND ----------

from azureml.core.compute import ComputeTarget

cluster_name = "NC6s-v3-SingleNode"
compute_target = ComputeTarget(workspace=ws, name=cluster_name)

# COMMAND ----------

import logging
from azureml.train.automl import AutoMLConfig
from azureml.automl.core.featurization import FeaturizationConfig

featurization_config = FeaturizationConfig(dataset_language='eng')

automl_settings = {
    "verbosity": logging.INFO,
    "enable_long_range_text": True,
    "enable_dnn": True,
    "experiment_timeout_minutes": 9900,
    "primary_metric": 'AUC_weighted'

}

automl_config = AutoMLConfig(
    task="text-classification",
    debug_log="automl_errors.log",
    compute_target=compute_target,
    training_data=ds_X_train,
    validation_data=ds_X_val,
    label_column_name="target",
    featurization=featurization_config,
    **automl_settings
)


# COMMAND ----------

experiment_name = "ownergroup-classification-automl"

experiment = Experiment(ws, experiment_name)

# COMMAND ----------

automl_run = experiment.submit(automl_config, show_output=False)
automl_run

# COMMAND ----------


