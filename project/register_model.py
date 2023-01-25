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

import azureml.core
from azureml.core import Workspace, Experiment, Environment, Model, Dataset, Run
from azureml.core.model import Model
from azureml.core.resource_configuration import ResourceConfiguration
from azureml.train.automl.run import AutoMLRun

#############################################
###  Find Best perorming Run for Pipeline
#############################################
def find_best_run(pipeline_run, metric_name, is_test, test_run_id):
    
    all_runs = pipeline_run.get_children(recursive=True)
    dic_runs = {}

    print('all runs {} '.format(all_runs))

    #####################################
    # Find all Run Steps with metric
    counter = 0
    for i, runstep in enumerate(all_runs):
        if is_test:
            if test_run_id in runstep.id:
                metrics = runstep.get_metrics()
                dic_runs[runstep.id] = {
                    'run': runstep,
                    'metrics': metrics
                }
                break
            else:
                continue
        else:
            metrics = runstep.get_metrics()
            if metric_name in metrics:
                dic_runs[runstep.id] = {
                    'run': runstep,
                    'metrics': metrics
                }
                print('Adding step {} and metrics: {}  '.format(runstep,metrics))
            counter+=1
    print(f'len(dic_runs) = {len(dic_runs)}')    
    print(f'Total number of valid runs: [{counter}]')

    #####################################
    # Find Run Step with best metric
    li_test_values = []
    best_performing_run = None

    for run_id in dic_runs:
        print('run_id', run_id)
        test_metric = dic_runs[run_id]['metrics'][metric_name]
        if (type(test_metric) == list):
            test_metric = float(test_metric[0])
        else:
            test_metric = float(test_metric)
        print(f'{metric_name} = {test_metric}')
        
        if len(li_test_values) == 0 or (len(li_test_values) > 0 and test_metric > max(li_test_values)):
            # if temporal_test_date == None:
            best_performing_run = dic_runs[run_id]

        # if temporal_test_date == None:
        li_test_values.append(test_metric)

    return best_performing_run, li_test_values   


#############################################
###  Find Datasets for best Run
#############################################
def find_run_datasets(run):
    ds_train = None
    ds_val = None
    ds_test = None

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

    return ds_train, ds_val, ds_test    
    
def register_automl_model(ws, best_run, model_name,  datasets, tags):
    #model_output = pipeline_run.get_pipeline_output("model_output")
    #print("Pipeline Data ==============: {}".format(model_output._path_on_datastore))

    best_exp = best_run.experiment

    automl_run = AutoMLRun(best_exp, run_id = best_run.id)
    print('AutoML Run ----------------{} -----------------'.format(automl_run))
    best_model_run = automl_run.get_best_child()

    print('best_model_run  {} '.format(best_model_run))
   
    model_output_dir = './model'

    os.makedirs(model_output_dir, exist_ok=True)
    best_model_run.download_files(prefix="outputs/", output_directory=model_output_dir,append_prefix=False)

    shutil.copy('score_automl.py', f'{model_output_dir}/score.py')
    shutil.copy('conda_env_automl.yml', './model/conda_env.yml')
    shutil.copy('sample_request_automl.json', './model/sample_request.json')
    print(' Dir {} '.format(os.listdir(model_output_dir)))
  
    # Register the Model
    model = Model.register(workspace=ws, 
                        datasets=datasets,
                        tags=tags,
                        model_name=model_name, 
                        resource_configuration=ResourceConfiguration(cpu=2, memory_in_gb=1),
                        model_path=model_output_dir)


def register_hyperdrive_model(ws,best_run, model_name, datasets, tags):
    # Download Best Model
    dir = f'output'

    isdir = os.path.isdir(dir)
    if isdir:
        shutil.rmtree(dir)

    model_directory = f'{dir}/outputs/model'
    os.makedirs(model_directory,exist_ok=True)
    print(f'the output path: [{model_directory}]')
    
    best_run.download_files(prefix="outputs/model", output_directory=dir, timeout_seconds=6000)

    shutil.copy('score.py', model_directory)
    shutil.copy('conda_env_automl.yml', f'{model_directory}/conda_env.yml')
    shutil.copy('sample_request.json', model_directory)
   
    num_labels = len(pdf_train[args.target_name].unique())
    print(f'Number of labels: [{num_labels}]')

    li_target = list(pdf_train[args.target_name].unique())
    with open(f"{model_directory}/target_list.json", "wb") as outfile:
        pickle.dump(li_target, outfile)

    # Register the Model
  
    model = Model.register(workspace=ws, 
                        datasets=datasets, 
                        tags=tags,
                        model_name=model_name, 
                        resource_configuration=ResourceConfiguration(cpu=2, memory_in_gb=1),
                        model_path=model_directory)

#############################################
###   Main
#############################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--is-test', type=int, dest='is_test', default=0, help='if is_test is passed as 1 then this is a short circuit job')
    parser.add_argument('--test-run-id', type=str, dest='test_run_id', default='HD_03ef2102-1cf6-49a7-84a7-140720e57834_2', help='if is_test is passed as 1 then this parameter is entertained')
    parser.add_argument('--metric-name', type=str, dest='metric_name', default='test_metric', help='the primary metric to find the best model')
    parser.add_argument('--model-name', type=str, dest='model_name', help='model name to be registered')
    parser.add_argument('--target-name', type=str, dest='target_name', default='target', help='target column name')
   
    args = parser.parse_args()

    print(f'args: [{args}]')
    
    is_test = args.is_test
    test_run_id = args.test_run_id
    metric_name = args.metric_name
    model_name = args.model_name

    run = Run.get_context()
    parent_id = run.parent.id
    ws = run.experiment.workspace
    exp = run.experiment

    print('Run {} will be used {}'.format(run, run.id))
    print('Experiment {} will be used'.format(exp))

    parent_id = run.parent.id
    pipeline_run = ws.get_run(parent_id)

    print('pipeline_run -- {} will be used'.format(pipeline_run))
   
    ### Find best run based on metric passed
    best_performing_run, li_test_values = find_best_run(pipeline_run, metric_name, is_test, test_run_id)
    if not best_performing_run:
        print('No run is found')
        os._exit(os.EX_OK)
     
    ##### Log metadada about best run 
    best_run = best_performing_run['run']
    print(f'Best performing run for [{metric_name}] = {max(li_test_values)}')
    print(f'run id: {best_run.id}')
    
    run.log('best_run_id', best_run.id)
    run.log(f'best {metric_name}', f'{best_performing_run["metrics"][metric_name]}')
    print(f'{metric_name}: {best_performing_run["metrics"][metric_name]}')
   
    ##### Check is Best Run is AUtoML
    is_automl_best=False
    if "AutoML" in best_run.name: 
       is_automl_best=True
       best_run = next(r for r in pipeline_run.get_children(recursive=True) if r.name == 'AutoML_Classification')
       print(f'!!! Best model found by AutoML !!!! ')
    else:
       print(f'!!! Best model found by HyperDrive !!!! ')

    print('best run_id', best_run.id) 
    run.log('is_automl_best', is_automl_best)
    
    #### Find  Best Run datasets and Tags
    ds_train, ds_val, ds_test = find_run_datasets(best_run)  
    pdf_train = ds_train.to_pandas_dataframe()
    print(f"Train dataset name: {ds_train.name}, V:{ds_train.version}")
    datasets=[('train dataset', ds_train),
                ('val dataset', ds_val)
             ]
    tags = {
        'run_id': best_run.id,
        '--metric-name': metric_name,
         '--model-name': model_name
    }
   
    if is_automl_best == True:
        register_automl_model(ws, best_run, model_name,  datasets, tags)
    else:
        register_hyperdrive_model(ws, best_run, model_name,  datasets, tags)

