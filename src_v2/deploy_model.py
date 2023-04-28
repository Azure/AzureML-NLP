import argparse
import datetime
import os
from os.path import exists

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)
from azure.ai.ml import MLClient
from azure.identity import ManagedIdentityCredential

def create_endpoint(ml_client, online_endpoint_name):
    # create an online endpoint
    endpoint = ManagedOnlineEndpoint(
        name=online_endpoint_name,
        description="this is the online endpoint for the sentiment classifier",
        auth_mode="key"
    )

    poller = ml_client.online_endpoints.begin_create_or_update(endpoint)
    poller.wait()
    return endpoint


def get_model_object(ml_client, model_name):
    model = list(ml_client.models.list(model_name))[0]

    return model


def create_deployment(ml_client, deploy_name, online_endpoint_name, model, env_name):
    env_list = list(ml_client.environments.list(name=env_name))
    env = env_list[0]

    deployment = ManagedOnlineDeployment(
        name=deploy_name,
        endpoint_name=online_endpoint_name,
        model=model,
        environment=env,
        code_configuration=CodeConfiguration(
            code='.', scoring_script="score.py"
        ),
        instance_type="Standard_E4s_v3",
        instance_count=1,
    )

    poller = ml_client.begin_create_or_update(deployment)
    poller.wait()

    return deployment


def test_deployment(online_endpoint_name, deploy_name):
    result_raw = ml_client.online_endpoints.invoke(
        endpoint_name=online_endpoint_name,
        deployment_name=deploy_name,
        request_file='sample_request.json')

    result = list(eval(result_raw))
    print(f'test result: {result}')

    return len(result) == 2


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
    parser.add_argument('--environment-name', type=str, dest='environment_name', help='environment name')
    parser.add_argument('--subscription-id', type=str, dest='subscription_id', help='subscription id')
    parser.add_argument('--resource-group', type=str, dest='resource_group', help='resource group')
    parser.add_argument('--workspace', type=str, dest='workspace', help='workspace name')
    parser.add_argument('--model-data', type=str, dest='model_data', help='workspace')
    
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    print(f'known arguments: {args}')
    print(f'unknown arguments: {unknown}')

    # get a handle to the workspace
    ml_client = MLClient(
        ManagedIdentityCredential(), args.subscription_id, args.resource_group, args.workspace
    )
    print(ml_client.subscription_id, ml_client.resource_group_name, ml_client.workspace_name, sep='\n')
    
    endpoint = create_endpoint(ml_client, args.endpoint_name)
    print(f'Endpoint [{args.endpoint_name}] is created')

    model = get_model_object(ml_client, args.model_name)

    deploy_name = "deployment-" + datetime.datetime.now().strftime("%m%d%H%M%f")

    print(f'Creation of deployment [{deploy_name}] is started')
    deployment = create_deployment(ml_client, deploy_name, args.endpoint_name, model, args.environment_name)
    print(f'Deployment [{deploy_name}] is created')

    is_success = test_deployment(args.endpoint_name, deploy_name)
    print(f'Testing the deployment is completed')
    
    if is_success:
        print(f'Testing of the deployment is SUCCESSFUL')
        endpoint.traffic = {deploy_name: 100}
        ml_client.begin_create_or_update(endpoint)
        print(f'Traffic allocation is set to 100% for deployment [{deploy_name}]')
        delete_old_deployments(args.endpoint_name, deploy_name)
    else:
        raise SystemExit("The test of the deployment was not successful")
