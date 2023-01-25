import os
import argparse
import shutil
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    Model,
)
from azure.identity import ManagedIdentityCredential

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, dest='model_path', help='Model path')
    parser.add_argument('--model-name', type=str, dest='model_name', help='model name to be registered')
    parser.add_argument('--subscription-id', type=str, dest='subscription_id', help='subscription id')
    parser.add_argument('--resource-group', type=str, dest='resource_group', help='resource group')
    parser.add_argument('--workspace', type=str, dest='workspace', help='workspace')
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
    print('list args.model_path', os.listdir(args.model_path))

    custom_model = Model(
        path=args.model_path,
        type=AssetTypes.CUSTOM_MODEL,
        name=args.model_name,
        description="Transformer based model build through AML V2",
    )

    ml_client.models.create_or_update(custom_model)


    os.makedirs(args.model_data, exist_ok=True)

    with open(f'{args.model_data}/temp.txt', 'w') as f:
        f.write('Create a dummy file !')

    with open(f'{args.model_data}/temp2.txt', 'w') as f:
        f.write('Create a dummy file !')
