# Accelerator - Fine-Tuning Language Model in AML with Custom Dataset

This work is aimed to provide an example to fine-tune a custom NLP model using native AML capabilities including HyperDrive to tune the hyperparameters and Managed Endpoint to deploy the model. This version is developed using AML V1 although the deployment of the model to Managed Endpoints is developed using SDK V2.

This work leverages [AML Pipelines V1](https://learn.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py&preserve-view=true). The model is trained using several models including [Bert Based Cased](https://huggingface.co/tftransformers/bert-base-cased) and is deployed in [Managed Endpoints using SDKV2](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-managed-online-endpoints). In order to run the pipeline, you need to run the [DefinePipeline.ipynb](./DefinePipeline.ipynb) notebook.

![Pipeline](./assets/Pipeline.jpg)

## Requirements
* **Dataset** - Amazon Dataset is used - the details are explained below.
* **AML Workspace** - This work is optimized for Azure Machine Learning, therefore you need to have an active Azure Subscription (free subscription works as well) and an Azure Machine Learning Workspace.
* **AML Computes** - The training requires a GPU based compute. Ideally, an [NC6_V3](https://learn.microsoft.com/en-us/azure/virtual-machines/ncv3-series) which contains an [NVidia Tesla V100](https://www.nvidia.com/en-gb/data-center/tesla-v100/). After the training step, a CPU based [AML Cluster](https://learn.microsoft.com/en-us/azure/machine-learning/v1/how-to-create-attach-compute-cluster?tabs=python#what-is-a-compute-cluster) could be used for registration and deployment steps.
* **Managed Endpoint** - The model is deployed as an online [Managed Endpoint](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-managed-online-endpoints?tabs=azure-cli) to serve incoming requests.

## Dataset
The dataset is the Amazon Review dataset from [UCSD](http://jmcauley.ucsd.edu/data/amazon/) archive to predict sentiment of the reviews from ratings. For this sample, the Automotive dataset is chosen which has around 19K samples. [DefinePipeline.ipynb](./DefinePipeline.ipynb) notebook contains the steps to download and prepare the dataset.

## AML Pipeline
This work is using AML Pipelines for easier operationalization of the workload. It consists of three steps which includes a HyperDrive Step to hyper tune and train the model, Model Registration step to add the model into the model repository, and finally a Model Deployment step to deploy the model as an online Managed Endpoint.

## Step 1 - HyperDrive_Step

This step is a [HyperDrive](https://learn.microsoft.com/en-us/python/api/azureml-pipeline-steps/azureml.pipeline.steps.hyperdrivestep?view=azure-ml-py) step which tunes a [HuggingFace Transformer Bert Base Cased](https://huggingface.co/tftransformers/bert-base-cased). The parameters of the step can help increasing the trials and test different combinations of hyperparameters to get to the best model.

The most important parameters are `learning-rate` and `epochs`. During tests, we learned that `learning-rate` 5.5e-5 can on this dataset. The `epoch` value is recommended to be set to 3 or 4 based on the Bert paper.

![Hyper Drive Trials](./assets/HyperDrive.jpg)

## Step 2 - Register_Best_Model

In this step, the aim is to find the best model to register based on the previous runs. This step focuses on the historical runs and picks up the highest performing runs based on the `metric-name` parameter.

## Step 3 - Deploy_Latest_Model

In this step, the recently registered model is deployed as an `AML Managed Endpoint`. The logics deploys a new `deployment` with the newly registered model and if a simple test is passed, the traffic is increased to 100% and other deployments are deleted.

## Rooms for improvement

1- **Leverage larger models**: `Bert-base-cased` is considered a moderate sized model. Leveraging larger language models theoretically should improve the model performance. Models such as `Bert-Large`, `DeBerta`, `DeBerta-Large`, etc.

2- **Add more rigid tests in the model registry step**: currently, every new model is compared only by the statistical tests generated during the training. There could be more tests to make sure that the model is trustable before registering and deploying the model.

3- **Add the Telemetries in App Insights**: currently the online model doesn't record any telemetries to App Insights. In the future, we will this functionality.

