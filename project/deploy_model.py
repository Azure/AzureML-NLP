from subprocess import run, PIPE, STDOUT
import pkg_resources

def run_cmd(cmd):
    ps = run(cmd, stdout=PIPE, stderr=STDOUT, shell=True, text=True)
    print(ps.stdout)

run_cmd('pip install azure-ai-ml==1.2.0')

import argparse
import datetime
import os
from os.path import exists

import azureml.core
from azureml.core import Run
from azureml.core.model import Model as Modelv1
from azureml.core import Workspace
from azureml.core import Environment as Environmentv1

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)
from azure.ai.ml import MLClient
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential 
from azure.identity import AzureCliCredential, ManagedIdentityCredential 

def create_endpoint(online_endpoint_name):
    # create an online endpoint
    endpoint = ManagedOnlineEndpoint(
        name=online_endpoint_name,
        description="this is the online endpoint for the sentiment classifier",
        auth_mode="key",
        kind="Managed"
    )

    poller = ml_client.online_endpoints.begin_create_or_update(endpoint)
    poller.wait()

    print("created Endpoint {}".format(endpoint))
    return endpoint


def get_model_object(model_name):
    reg_model = Modelv1(ws, name=model_name)

    dir = 'outputs'
    reg_model.download(target_dir=dir, exist_ok=True)

    prefix_path = "model"
    model_directory = f'{dir}/{prefix_path}'
    print(' Get Model object Dir {} '.format(os.listdir(model_directory)))

    model = Model(name=model_name, path=model_directory)
    return model, model_directory


def create_deployment(deploy_name, online_endpoint_name, model, model_directory):
    env = Environment(
        #image="mcr.microsoft.com/azureml/curated/azureml-automl-dnn-text-gpu:56",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        conda_file=f"{model_directory}/conda_env.yml"
    )
    
    if not exists(f'{model_directory}/score.py'):
        model_directory = ''

    print(' Create deployment Dir {} '.format(os.listdir(model_directory)))

    properties = {
        "endpoint_compute_type": "Managed"
    }

    deployment = ManagedOnlineDeployment(
        name=deploy_name,
        endpoint_name=online_endpoint_name,
        model=model,
        environment=env,
        code_configuration=CodeConfiguration(
            code=model_directory, scoring_script="score.py"
        ),
        instance_type="Standard_DS2_v2",
        instance_count=1,
        properties = properties
    )

    poller = ml_client.begin_create_or_update(deployment)
    
    poller.wait()
    print(ml_client.online_deployments.get_logs(
          name=deploy_name, endpoint_name=online_endpoint_name,  lines=50))          
    return deployment


def test_deployment(online_endpoint_name, deploy_name, sample_file):
    result_raw = ml_client.online_endpoints.invoke(
        endpoint_name=online_endpoint_name,
        deployment_name=deploy_name,
        request_file=sample_file)

    print(f'test result raw: {result_raw}')
    result = list(eval(result_raw))
    print(f'test result: {result}')

    return len(result) > 0


def delete_old_deployments(online_endpoint_name, skip_deploy_name):
    li_to_be_deleted_dep = []
    for online_deployment in ml_client.online_deployments.list(online_endpoint_name):
        if skip_deploy_name != online_deployment.name:
            li_to_be_deleted_dep.append(online_deployment.name)

    print(f'Deleting deployments: [{li_to_be_deleted_dep}]')
    for online_deployment in li_to_be_deleted_dep:
        ml_client.online_deployments.begin_delete(name=online_deployment, endpoint_name=online_endpoint_name)
    print(f'Deletion of the deployments is started for: [{li_to_be_deleted_dep}]')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint-name', type=str, dest='endpoint_name', help='Name of the endpoint')
    parser.add_argument('--model-name', type=str, dest='model_name', help='Name of the registered model')
    
    (args, extra_args) = parser.parse_known_args()

    print(f'args: {args}')
    print(f'extra_args: {extra_args}')

    run = Run.get_context()
    parent_id = run.parent.id
    
    ws = run.experiment.workspace
    exp = run.experiment
    

    ml_client = MLClient(
         ManagedIdentityCredential (), ws.subscription_id, ws.resource_group, ws.name
    )

    endpoint = create_endpoint(args.endpoint_name)
    print(f'Endpoint [{args.endpoint_name}] is created')

    model, model_directory = get_model_object(args.model_name)

    deploy_name = "deployment-" + datetime.datetime.now().strftime("%m%d%H%M%f")

    print(f'Creation of deployment [{deploy_name}] is started')
    print(f'Model {model}')
    print(f'Model {model_directory}')

    deployment = create_deployment(deploy_name, args.endpoint_name, model, model_directory)
    print(f'Deployment [{deploy_name}] is created')

    print(f'Testing the deployment with sample {model_directory}/sample_request.json')
    is_success = test_deployment(args.endpoint_name, deploy_name,f'{model_directory}/sample_request.json')
    print(f'Testing the deployment is completed')
    
    if is_success:
        print(f'Testing of the deployment is SUCCESSFUL')
        endpoint.traffic = {deploy_name: 100}
        ml_client.begin_create_or_update(endpoint)
        print(f'Traffic allocation is set to 100% for deployment [{deploy_name}]')

        delete_old_deployments(args.endpoint_name, deploy_name)
    else:
        raise SystemExit("The test of the deployment was not successful")