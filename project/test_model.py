import argparse
import json
import os
import sys
import shutil
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing
from datasets import  load_metric

import azureml.core
from azureml.core import Workspace, Experiment, Environment, Model, Dataset, Run
from azureml.core.model import Model
from azureml.core.resource_configuration import ResourceConfiguration
from azureml.train.automl.run import AutoMLRun

from transformers import TrainingArguments, Trainer, AutoTokenizer


def get_encode_labels(pdf, text_field_name):
    le = preprocessing.LabelEncoder()

    le.fit(list(pdf[text_field_name].unique()))
    
    return le

def adjust_tokenizer(model, tokenizer, new_tokens):
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

def tokenize_function(example, text_field_name, tokenizer):
    tokenized_batch = tokenizer(example[text_field_name], truncation=True)
    return tokenized_batch

def generate_tokenized_dataset(pdf, fields, le, target_name, text_field_name, tokenizer):
    from datasets import Dataset

    pdf['labels'] = le.transform(pdf[target_name])
    # pdf['labels'] = pdf[target_name]
    
    ds = Dataset.from_pandas(pdf[fields].dropna())
    
    tokenized_dataset = ds.map(tokenize_function, batched=True, fn_kwargs={"text_field_name": text_field_name, "tokenizer": tokenizer})
    return ds, tokenized_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric-name', type=str, dest='metric_name', default='test_f1_weighted', help='the primary metric to find the best model')
    parser.add_argument('--target-name', type=str, dest='target_name', default='target', help='target column name')
    parser.add_argument('--model-data', type=str, dest='model-data', default='target', help='best model data')
    parser.add_argument('--test_dataset', type=Dataset, dest='test_dataset', default='target', help='test dataset')
    parser.add_argument('--text-field-name', type=str, dest='text_field_name', default='reviewText', help='text field name')
 
    args = parser.parse_args()

    print(f'args: [{args}]')
    
    text_field_name = args.text_field_name
    target_name = args.target_name
    metric_name = args.metric_name

    run = Run.get_context()
    parent_id = run.parent.id
    ws = run.experiment.workspace
    exp = run.experiment


    print('Run {} will be used {}'.format(run, run.id))
    print('Experiment {} will be used'.format(exp))

    # ### Get Model
    pipeline_run = ws.get_run(parent_id)
    print('pipeline_run {} will be used'.format(pipeline_run))

    automl_run_found = next(r for r in pipeline_run.get_children(recursive=True) if r.name == 'AutoML_Classification')
    print('AutoML {}'.format(automl_run_found))

    automl_run = AutoMLRun(exp, run_id = automl_run_found.id)

   
    best_run, fitted_model = automl_run.get_output(metric=metric_name)
    print(best_run)
    print(fitted_model)

    # ### Prepare Data
    print('Tokenizing')
    test_dataset = Run.get_context().input_datasets['test_dataset']
    pdf_test = test_dataset.to_pandas_dataframe()
    print('Test Dataset {}'.format(pdf_test))

    new_tokens = []
    num_labels = len(pdf_test[target_name].unique())
    print(f'num_labels: {num_labels}')

    le = get_encode_labels(pdf_test, target_name)
    fields = [text_field_name, target_name, 'labels']
    print(f'le: {le}')
    print(f'fields: {fields}')

    test_ds, tokenized_test_ds = generate_tokenized_dataset(pdf_test, fields, le, target_name, text_field_name, fitted_model.tokenizer)
    
    print('Tokenized data is generated')


    # ### Training and metrics
    ## https://medium.com/nlplanet/bert-finetuning-with-hugging-face-and-training-visualizations-with-tensorboard-46368a57fc97

    test_predictions = fitted_model.predict(pdf_test)
    #print(f'predictions: {test_predictions}')
    
    # ### Calculate metrics
    test_references = test_ds[target_name]

    # Compute AUC_weighted specifically
    metric = load_metric("roc_auc", average='weighted')
    final_score = metric.compute(prediction_scores=test_predictions, references=test_references)
    
    print(f'Metrics:', final_score["roc_auc"])
    run.log(f'test_{metric_name}', f'{final_score["roc_auc"]}')  
