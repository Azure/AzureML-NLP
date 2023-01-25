import numpy as np
import pandas as pd
import argparse
import os
import re
import time
import glob
import joblib
import torch
import mlflow
import pickle
import shutil

from nvitop import ResourceMetricCollector
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn import preprocessing
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import EarlyStoppingCallback
from transformers.integrations import MLflowCallback
from transformers import AutoTokenizer, DataCollatorWithPadding

from mlflow.tracking import MlflowClient

def get_model(base_checkpoint, num_labels):
    model = AutoModelForSequenceClassification.from_pretrained(base_checkpoint, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(base_checkpoint)
    
    return model, tokenizer

def adjust_tokenizer(model, tokenizer, new_tokens):
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

def get_tokens(pdf = None):
    new_tokens = []

    return new_tokens

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='macro')
    precision = precision_score(y_true=labels, y_pred=pred, average='macro')
    f1 = f1_score(y_true=labels, y_pred=pred, average='macro')
    auc = roc_auc_score(y_true=labels, y_score=pred, average='macro')

    recall_weighted = recall_score(y_true=labels, y_pred=pred, average='weighted')
    precision_weighted = precision_score(y_true=labels, y_pred=pred, average='weighted')
    f1_weighted = f1_score(y_true=labels, y_pred=pred, average='weighted')
    auc_weighted = roc_auc_score(y_true=labels, y_score=pred, average='weighted')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "auc": auc,
            "recall_weighted": recall_weighted, "precision_weighted": precision_weighted, "f1_weighted": f1_weighted, "auc_weighted": auc_weighted}

def get_encode_labels(pdf, text_field_name):
    le = preprocessing.LabelEncoder()

    le.fit(list(pdf[text_field_name].unique()))
    
    return le

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

def get_datasets(is_local, is_test, is_final):
    if is_local:
        print('the job is running locally')
        pdf_train = pd.read_parquet('../data/pdf_train.parquet')
        pdf_validation = pd.read_parquet('../data/pdf_validation.parquet')
        pdf_test = pd.read_parquet('../data/pdf_test.parquet')

        new_tokens = get_tokens()
    else:
        from azureml.core import Dataset
        
        pdf_train = pd.read_csv(training_dataset)
        pdf_validation = pd.read_csv(val_dataset)
        pdf_test = pd.read_csv(test_dataset)
        

    if is_test:
        print('the job is a test job')
        _, pdf_train = train_test_split(pdf_train, test_size=4000/pdf_train.shape[0], stratify=pdf_train[args.target_name])
        # _, pdf_validation = train_test_split(pdf_validation, test_size=4000/pdf_validation.shape[0], stratify=pdf_validation[args.target_name])
        # _, pdf_test = train_test_split(pdf_test, test_size=4000/pdf_test.shape[0], stratify=pdf_test[args.target_name])

    if is_final:
        pdf_train = pd.concat([pdf_train, pdf_validation, pdf_test])
    
    new_tokens = get_tokens(pdf_train)


    print(f'pdf_train is imported with "{pdf_train.shape}" rows')
    print(f'pdf_validation is imported with "{pdf_validation.shape}" rows')
    print(f'pdf_test is imported with "{pdf_test.shape}" rows')
    
    return pdf_train, pdf_validation, pdf_test, new_tokens

def test_model(trainer, ds, prefix):
    test_result = trainer.predict(ds)

    metrics = test_result.metrics.keys()
    # print(f'len(metrics): {metrics}')
    
    for m in metrics:
        metric_name = f'{prefix}_{m.replace("test_", "")}'
        metric_value = test_result.metrics[m]
        
        print(metric_name, metric_value)
        mlflow.log_metric(metric_name, metric_value)

    return test_result

def on_collect(metrics):
    try:
        if not mlflow.active_run().info.run_id:
            return True

        gpu_metrics = {}

        for key in metrics.keys():
            if 'mean' in key and not 'load_average' in key and not 'fan_speed' in key and 'pid' not in key:
                symbole = ' %' if '%' in key else ''
                key_name = key.split(' ')[0] + symbole
                value = round(metrics[key], 2) if metrics[key] else 0
                if 'gpu' in key:
                    # gpu_metrics[key_name] = value
                    gpu_name_g = re.search(r'/gpu:\d+/', key_name)
                    gpu_name = gpu_name_g.group()
                    
                    if gpu_name in gpu_metrics:
                        gpu_metrics[gpu_name][key_name] = value
                    else:
                        gpu_metrics[gpu_name] = {
                            key_name: value
                        }
                else:
                    mlflow.log_metric(key_name, value)
        
        for gpu_name in gpu_metrics.keys():
            mem_vals = {}
            perc_vals = {}
            for metrics_name in gpu_metrics[gpu_name]:
                if '%' in metrics_name:
                    perc_vals[metrics_name] = gpu_metrics[gpu_name][metrics_name]
                elif 'memory' in metrics_name:
                    mem_vals[metrics_name] = gpu_metrics[gpu_name][metrics_name]
                else:
                    mlflow.log_metric(key_name, value)

            mlflow.log_metric(f"{gpu_name} utilization", **perc_vals)
            mlflow.log_metric(f"{gpu_name} memory", **mem_vals)
                
    except Exception as exp:
        print(exp)
        
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-checkpoint', type=str, dest='base_checkpoint', default='bert-base-uncased', help='base model name')
    parser.add_argument('--training-dataset', type=str, dest='training_dataset', help='Training dataset')
    parser.add_argument('--val-dataset', type=str, dest='val_dataset', help='Validation dataset')
    parser.add_argument('--test-dataset', type=str, dest='test_dataset', help='Test dataset')
    parser.add_argument('--model-path', type=str, dest='model_path', help='Model path')
    parser.add_argument('--target-name', type=str, dest='target_name', default='target', help='target column name')
    parser.add_argument('--text-field-name', type=str, dest='text_field_name', default='text_field', help='text field name')
    parser.add_argument('--is-test', type=int, dest='is_test', default=0, help='1 if this is a test run')
    parser.add_argument('--is-final', type=int, dest='is_final', default=0, help='1 if this is a final run (train on all data)')
    parser.add_argument('--is-local', type=int, dest='is_local', default=0, help='1 if this is a local run')
    parser.add_argument('--is-jump', type=int, dest='is_jump', default=0, help='by passes the entire step by minimal operations')
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=16, help='training and validation batch size')
    parser.add_argument('--no-epochs', type=int, dest='no_epochs', default=3, help='no of epochs')
    parser.add_argument('--learning-rate', type=float, dest="learning_rate", default=5e-5, help='setting learning-rate for the TrainingArguments')
    parser.add_argument('--warmup-steps', type=int, dest="warmup_steps", default=0, help='setting warmup-steps for the TrainingArguments')
    parser.add_argument('--weight-decay', type=float, dest="weight_decay", default=0.0, help='setting weight-decay for the TrainingArguments')
    parser.add_argument('--adam-beta1', type=float, dest="adam_beta1", default=0.9, help='setting adam-beta1 for the TrainingArguments')
    parser.add_argument('--adam-beta2', type=float, dest="adam_beta2", default=0.999, help='setting adam-beta2 for the TrainingArguments')
    parser.add_argument('--adam-epsilon', type=float, dest="adam_epsilon", default=1e-8, help='setting adam-epsilon for the TrainingArguments')
    parser.add_argument('--evaluation-strategy', type=str, dest='evaluation_strategy', default='epoch', help='evaluation strategy')
    parser.add_argument('--collect-resource-utilization', type=int, dest='collect_resource_utilization', default=1, help='whether or not to collect granular resource utilization as metrics')
    parser.add_argument('--resource-utilization-interval', type=float, dest='resource_utilization_interval', default=5.0, help='the interval (in seconds) in which the resource utilization is collected')

    # run = Run.get_context()
    mlflow.start_run()

    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    print(f'known arguments: {args}')
    print(f'unknown arguments: {unknown}')

    base_checkpoint = args.base_checkpoint
    training_dataset = args.training_dataset
    val_dataset = args.val_dataset
    test_dataset = args.test_dataset
    model_path = args.model_path
    text_field_name = args.text_field_name
    target_name = args.target_name
    batch_size = args.batch_size
    is_test = args.is_test
    is_final = args.is_final
    is_local = args.is_local
    is_jump = args.is_jump
    no_epochs = args.no_epochs
    learning_rate = args.learning_rate
    warmup_steps = args.warmup_steps
    weight_decay = args.weight_decay
    adam_beta1 = args.adam_beta1
    adam_beta2 = args.adam_beta2
    adam_epsilon = args.adam_epsilon
    evaluation_strategy = args.evaluation_strategy
    collect_resource_utilization = args.collect_resource_utilization
    resource_utilization_interval = args.resource_utilization_interval

    if collect_resource_utilization == 1:
        collector = ResourceMetricCollector(interval=resource_utilization_interval)
        daemon = collector.daemonize(on_collect, interval=None, tag="")

    # print(f"run.input_datasets [{run.input_datasets}]")
    
    if is_test:
        no_epochs = 1

    print(f'base_checkpoint: {base_checkpoint}')
    print(f'training_dataset: {training_dataset}')
    print(f'val_dataset: {val_dataset}')
    print(f'test_dataset: {test_dataset}')
    print(f'model_path: {model_path}')
    print(f'text_field_name: {text_field_name}')
    print(f'target_name: {target_name}')
    print(f'batch_size: {batch_size}')
    print(f'is_test: {is_test}')
    print(f'is_final: {is_final}')
    print(f'is_local: {is_local}')
    print(f'is_jump: {is_jump}')
    print(f'no_epochs: {no_epochs}')
    print(f'learning_rate: {learning_rate}')
    print(f'warmup_steps: {warmup_steps}')
    print(f'weight_decay: {weight_decay}')
    print(f'adam_beta1: {adam_beta1}')
    print(f'adam_beta2: {adam_beta2}')
    print(f'adam_epsilon: {adam_epsilon}')
    print(f'evaluation_strategy: {evaluation_strategy}')
    print(f'evaluation_strategy: {evaluation_strategy}')
    print(f'collect_resource_utilization: {collect_resource_utilization}')
    print(f'resource_utilization_interval: {resource_utilization_interval}')

    # model_path = 'outputs/model'
    if is_jump == 1:
        os.makedirs(model_path, exist_ok=True)

        with open(f'{model_path}/temp.txt', 'w') as f:
            f.write('Create a dummy file !')

        with open(f'{model_path}/temp2.txt', 'w') as f:
            f.write('Create a dummy file !')

        mlflow.log_metric('test_f1_weighted', 0.5)

        os._exit(os.EX_OK)

    pdf_train, pdf_validation, pdf_test, new_tokens = get_datasets(is_local, is_test, is_final)

    num_labels = len(pdf_train[target_name].unique())
    print(f'num_labels: {num_labels}')

    model, tokenizer = get_model(base_checkpoint, num_labels)
    model, tokenizer = adjust_tokenizer(model, tokenizer, new_tokens)

    le = get_encode_labels(pdf_train, target_name)

    fields = [text_field_name, target_name, 'labels']

    train_ds, tokenized_train_ds = generate_tokenized_dataset(pdf_train, fields, le, target_name, text_field_name, tokenizer)
    validation_ds, tokenized_validation_ds = generate_tokenized_dataset(pdf_validation, fields, le, target_name, text_field_name, tokenizer)
    test_ds, tokenized_test_ds = generate_tokenized_dataset(pdf_test, fields, le, target_name, text_field_name, tokenizer)
    # temporal_test_ds, tokenized_temporal_test_ds = generate_tokenized_dataset(pdf_temporal_test, fields, le, target_name, text_field_name, tokenizer)

    print('Tokenized data is generated')

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir="outputs",
        evaluation_strategy=evaluation_strategy, # "steps", # "epoch"
        save_strategy=evaluation_strategy,
        eval_steps=500,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=no_epochs,
        seed=0,
        load_best_model_at_end=True,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        warmup_steps=warmup_steps,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_validation_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # trainer.callback_handler.remove_callback(MLflowCallback)

    # Train pre-trained model
    print("Training started")
    trainer.train()
    print("Training is finished")
    # mlflow.pytorch.save_model(trainer, model_path)
    trainer.save_model(model_path)

    # to save encoder 
    joblib.dump(le, f'{model_path}/labelEncoder.joblib',compress=9)
    li_target = list(pdf_train[target_name].unique())
    with open(f"{model_path}/target_list.json", "wb") as outfile:
        pickle.dump(li_target, outfile)

    shutil.copy('score.py', model_path)

    print(f"Model and the assets are saved to {model_path}")

    print("Evaluation is started")
    trainer.evaluate()
    print("Evaluation is completed")

    print("Test on training data is started")
    test_model(trainer, tokenized_train_ds, 'train')
    print("Test on training data is completed")

    print("Test is started")
    test_model(trainer, tokenized_test_ds, 'test')
    print("Test is completed")

    mlflow.end_run()

    # print("Temporal test is started")
    # test_model(trainer, tokenized_temporal_test_ds, 'temporal_test')
    # print("Temporal test is completed")

    # train_ds, tokenized_train_ds = generate_tokenized_dataset(pdf_train[['OTHER']], fields, le, target_name, text_field_name, tokenizer)
    # validation_ds, tokenized_validation_ds = generate_tokenized_dataset(pdf_validation, fields, le, target_name, text_field_name, tokenizer)
    # pdf_test_no_other = pdf_test[pdf_test[args.target_name] != "OTHER"]
    # print(f'size of pdf_test without [OTHER class]: "{pdf_test_no_other.shape[0]}"')
    # pdf_temporal_test_no_other = pdf_temporal_test[pdf_temporal_test[args.target_name] != "OTHER"]
    # print(f'size of pdf_temporal_test without [OTHER class]: "{pdf_temporal_test_no_other.shape[0]}"')

    # test_ds, tokenized_test_ds_no_other = generate_tokenized_dataset(pdf_test_no_other, fields, le, target_name, text_field_name, tokenizer)
    # temporal_test_ds, tokenized_temporal_test_ds_no_other = generate_tokenized_dataset(pdf_temporal_test_no_other, fields, le, target_name, text_field_name, tokenizer)

    # print("Test (no other class) is started")
    # test_model(trainer, tokenized_test_ds_no_other, 'test_no_other')
    # print("Test (no other class) is completed")
    
    # print("Temporal test (no other class) is started")
    # test_model(trainer, tokenized_temporal_test_ds_no_other, 'temp_test_no_ot')
    # print("Temporal test (no other class) is completed")


