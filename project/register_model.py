import argparse
import json
import os
import joblib
import shutil
import os
import sys
import shutil
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, recall_score, precision_score, f1_score

import azureml.core
from azureml.core import Workspace, Experiment, Environment, Model, Dataset, Run
from azureml.core.model import Model
from azureml.core.resource_configuration import ResourceConfiguration

parser = argparse.ArgumentParser()
parser.add_argument('--is-test', type=int, dest='is_test', default=0, help='if is_test is passed as 1 then this is a short circuit job')
parser.add_argument('--test-run-id', type=str, dest='test_run_id', default='HD_03ef2102-1cf6-49a7-84a7-140720e57834_2', help='if is_test is passed as 1 then this parameter is entertained')
parser.add_argument('--metric-name', type=str, dest='metric_name', default='test_f1_weighted', help='the primary metric to find the best model')
parser.add_argument('--second-metric', type=str, dest='second_metric', default='test_f1', help='the secondary metric to report on the best model')
parser.add_argument('--model-name', type=str, dest='model_name', help='model name to be registered')
parser.add_argument('--target-name', type=str, dest='target_name', default='target', help='target column name')

args = parser.parse_args()

print(f'args: [{args}]')

run = Run.get_context()
parent_id = run.parent.id
ws = run.experiment.workspace
exp = run.experiment

counter = 0
best_temporal_f1_weighted = 0.0
is_test = args.is_test
test_run_id = args.test_run_id


all_runs = exp.get_runs(include_children=True)
dic_runs = {}

print('Run {} will be used {}'.format(run, run.id))
print('Experiment {} will be used'.format(exp))

for i, run in enumerate(all_runs):
    if is_test:
        if test_run_id in run.id:
            metrics = run.get_metrics()
            dic_runs[run.id] = {
                'run': run,
                'metrics': metrics
            }
            break
        else:
            continue
    else:
        metrics = run.get_metrics()
        if args.metric_name in metrics:
            dic_runs[run.id] = {
                'run': run,
                'metrics': metrics
            }
        counter+=1
print(f'len(dic_runs) = {len(dic_runs)}')

metric_name = args.metric_name
second_metric = args.second_metric
# temporal_test_date = args.temporal_test_date

li_test_values = []
best_performing_run = None

for run_id in dic_runs:
    print('run_id', run_id)
    test_f1_weighted = dic_runs[run_id]['metrics'][metric_name]
    if (type(test_f1_weighted) == list):
        test_f1_weighted = float(test_f1_weighted[0])
    else:
        test_f1_weighted = float(test_f1_weighted)
    print(f'{metric_name} = {test_f1_weighted}')
    
    if len(li_test_values) == 0 or (len(li_test_values) > 0 and test_f1_weighted > max(li_test_values)):
        # if temporal_test_date == None:
        best_performing_run = dic_runs[run_id]

    # if temporal_test_date == None:
    li_test_values.append(test_f1_weighted)

    # if temporal_test_date:
    #     dataset = dic_runs[run_id]['run'].get_details()['inputDatasets'][0]
    #     if 'temporal_test_date' in dataset['dataset'].tags:
    #         if dataset['dataset'].tags['temporal_test_date'] == temporal_test_date:
    #             # best_performing_run = dic_runs[run_id]
    #             li_test_values.append(test_f1_weighted)

ds_train = None
ds_val = None
ds_test = None

if not best_performing_run:
    print('No run is found')
else:
    run = best_performing_run['run']

    print(f'Best performing run for [{metric_name}] = {max(li_test_values)}')

    ds_train = None
    temporal_dataset = None

    print(f"run.get_details()['inputDatasets']: [{run.get_details()['inputDatasets']}]")
    for dataset in run.get_details()['inputDatasets']:
        print(f'dataset: {dataset}')
        print(f"dataset['dataset']: {dataset['dataset']}")
        print(f"type(dataset['dataset']): {type(dataset['dataset'])}")
        print(f"dataset['dataset'].name: {dataset['dataset'].name}")
        print(f"dataset['dataset'].version: {dataset['dataset'].version}")

        
        if dataset['dataset'].name == 'train_set':
            ds_train = dataset['dataset']

        if dataset['dataset'].name == 'val_set':
            ds_val = dataset['dataset']

        if dataset['dataset'].name == 'test_set':
            ds_test = dataset['dataset']
        # elif dataset['dataset'].name == 'owner_g_classfication_temporal_test':
        #     temporal_dataset = dataset

    print(f'run id: {run.id}')
    # print(f'Temporal test date: {temporal_test_date}')
    print(f'{metric_name}: {best_performing_run["metrics"][metric_name]} - {second_metric}: {best_performing_run["metrics"][second_metric]}')
    print(f"Train dataset name: {ds_train.name}, V:{ds_train.version}")
    # print(f'Train dataset name: {temporal_dataset["dataset"].name}, V:{temporal_dataset["dataset"].version}')

    run.log('run_id', run.id)
    # run.log('Temporal_test_date', temporal_test_date)
    run.log(f'best {metric_name}', f'{best_performing_run["metrics"][metric_name]}')
    run.log(f'best {second_metric}', f'{best_performing_run["metrics"][second_metric]}')
    run.log('Train_dataset_name', f'{ds_train.name}, V:{ds_train.version}')
    # run.log('Train_dataset_name', f'{temporal_dataset["dataset"].name}, V:{temporal_dataset["dataset"].version}')

print(f'Total number of valid runs: [{counter}]')

if not best_performing_run:
    os._exit(os.EX_OK)

# ds_temporal_test = Dataset.get_by_name(ws, name="owner_g_classfication_temporal_test", version=temporal_dataset["dataset"].version)

pdf_train = ds_train.to_pandas_dataframe()

run = best_performing_run['run']

dir = f'output'

isdir = os.path.isdir(dir)
if isdir:
    shutil.rmtree(dir)

run.download_files(prefix="outputs/model", output_directory=dir, timeout_seconds=6000)

model_directory = f'{dir}/outputs/model'
print(f'the output path: [{model_directory}]')
shutil.copy('score.py', model_directory)

num_labels = len(pdf_train[args.target_name].unique())
print(f'Number of labels: [{num_labels}]')

li_target = list(pdf_train[args.target_name].unique())
with open(f"{model_directory}/target_list.json", "wb") as outfile:
    pickle.dump(li_target, outfile)

# model = AutoModelForSequenceClassification.from_pretrained(model_directory, num_labels=num_labels)
# tokenizer = AutoTokenizer.from_pretrained(model_directory)
# le=joblib.load(model_directory + '/labelEncoder.joblib')
# print('Model objects and their dependencies are loaded')

tags = {
    'run_id': run.id,
    '--metric-name': args.metric_name,
    '--second-metri': args.second_metric,
    # '--temporal-test-date': args.temporal_test_date,
    '--model-name': args.model_name
}

model = Model.register(workspace=ws, 
                       datasets=[('train dataset', ds_train), 
                                 ('val dataset', ds_val),
                                 ('test dataset', ds_test),
                                 # ('temporal_test dataset', ds_temporal_test)
                                 ], 
                       tags={'run_id': run.id},
                       # description="Service Desk Concierge Model",
                       model_name=args.model_name, 
                       resource_configuration=ResourceConfiguration(cpu=2, memory_in_gb=1),
                       model_path=model_directory)