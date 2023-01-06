import os
import logging
import json
import numpy
import joblib
import sys
import shutil
import pickle
import pandas as pd
import numpy as np

import torch
from transformers import AutoTokenizer, Trainer, AutoModelForSequenceClassification, AutoTokenizer


def load_model():
    model_directory = os.getenv("AZUREML_MODEL_DIR") + '/model'
    print(f'the output path: [{model_directory}]')

    if os.path.exists(model_directory):
        print(f'os.listdir: [{os.listdir(model_directory)}]')
    else:
        print(f'[{model_directory}] doesn\'t exist')

    with open(f"{model_directory}/target_list.json", "rb") as outfile:
        li_target = pickle.load(outfile)

    num_labels = len(li_target)  # len(pdf[target_field].unique())
    print(f'Number of labels: [{num_labels}]')

    model = AutoModelForSequenceClassification.from_pretrained(
        model_directory, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    le = joblib.load(model_directory + '/labelEncoder.joblib')

    print('Model objects and their dependencies are loaded')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    model.zero_grad()
    print(device)
    trainer = Trainer(model=model, tokenizer=tokenizer)

    return trainer, tokenizer, le


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    global le
    global tokenizer

    model, tokenizer, le = load_model()

    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    logging.info("Init complete")


def tokenize_function(example):
    tokenized_batch = tokenizer(example['text'], truncation=True)
    return tokenized_batch


def generate_tokenized_dataset(data):
    from datasets import Dataset
    # pdf['labels'] = pdf[target_name]

    pdf = pd.DataFrame(data)
    print('pdf')
    print(pdf)
    pdf_cleaned = pdf.dropna()
    print(f'pdf_cleaned: {pdf_cleaned}')
    ds = Dataset.from_pandas(pdf_cleaned)

    tokenized_dataset = ds.map(tokenize_function, batched=True)
    return tokenized_dataset


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info("Request received")
    print(f'raw data: {raw_data}')
    data = json.loads(raw_data)["data"]
    print(f'data: {data}')

    tokenized_data = generate_tokenized_dataset(data)
    result = model.predict(tokenized_data)

    pred = np.argmax(result.predictions, axis=1)
    print(f'pred: {pred}')

    result_enc = le.inverse_transform(pred)
    print(f'result_enc: {result_enc}')
    logging.info("Request processed")

    return result_enc.tolist()