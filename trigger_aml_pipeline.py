# Databricks notebook source
# MAGIC %md
# MAGIC # Trigger ECD horizon AML pipeline
# MAGIC **TODO** 2022-09-20: This notebook doesn't actually trigger the pipeline right now. It is commented out until the AML pipeline is updated to actually retrain on the new data.
# MAGIC 
# MAGIC Presently, the AML pipeline will continue to do a "temporal test" split at September 2022, so any new data would only be used in the test set.
# MAGIC 
# MAGIC The pipeline must be updated to have a "--is-final" option which combines the train and test sets during training. Then, running the pipeline with "--is-final" will use all of the new data.
# MAGIC 
# MAGIC The pipeline publishing step also needs to be updated. All of this is documented in the DefinePipeline.ipynb (recommend to edit in AzureML VScode integration).
# MAGIC 
# MAGIC After that is done, write and test the new code in the last cell of this notebook to trigger the latest version of the published pipeline.

# COMMAND ----------

!pip install azureml-sdk

# COMMAND ----------

# Workaround for import error in Azure ML SDK:
# https://github.com/Azure/azure-sdk-for-python/issues/23697
import typing_extensions
from importlib import reload
reload(typing_extensions)

# Actual imports
import azureml.core
from azureml.core import Workspace, Environment, Experiment, Datastore, Dataset, ScriptRunConfig, Run
from azureml.pipeline.core import Pipeline, PipelineData, TrainingOutput, PublishedPipeline, PipelineEndpoint
from azureml.core.authentication import ServicePrincipalAuthentication

# COMMAND ----------

# Our account details
workspace_name = "ScScCPS-DSAI-Lab-dev-mlw"
resource_group = "ScSc-DSAI-Lab-dev-rg"
kv_name = "ScScCSV-DSAI-Lab-dev-kv"

sp_pw = dbutils.secrets.get(scope=kv_name, key="service-principal-password")
tenant_id = dbutils.secrets.get(scope=kv_name, key="tenant-id")
sp_id = dbutils.secrets.get(scope=kv_name, key="service-principal-id")
subscription_id = dbutils.secrets.get(scope=kv_name, key="subscription-id")

svc_pr = ServicePrincipalAuthentication(tenant_id=tenant_id, service_principal_id=sp_id, service_principal_password=sp_pw)

ws = Workspace(
        workspace_name = workspace_name,
        subscription_id = subscription_id,
        resource_group = resource_group, 
        auth = svc_pr
    )

# COMMAND ----------

print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')

# COMMAND ----------

published_pipeline = None

published_pipelines = PublishedPipeline.list(ws)

ECD_pipelines = []

for published_pipeline in  published_pipelines:
    if published_pipeline.name.startswith('ECD-horizon'):
        ECD_pipelines.append(published_pipeline)
        
ECD_pipelines = sorted(ECD_pipelines, key=lambda x:x.name, reverse=True)

print(type(ECD_pipelines[0]))
        
if len(ECD_pipelines) > 0:
    experiment = Experiment(workspace=ws, name='transformer_hp')
#     pipeline_run = experiment.submit(ECD_pipelines[0].id)
    published_pipeline = ECD_pipelines[0]
    published_pipeline.submit(ws, experiment.name, pipeline_parameters=None, _workflow_provider=None, _service_endpoint=None, parent_run_id=None, continue_on_step_failure=None)

# COMMAND ----------


