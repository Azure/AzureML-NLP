import os
import sys
import joblib
import shutil
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from IPython.core.display import HTML

import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, Trainer, AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(os.path.join(os.path.join(os.getcwd(), ".."), 'project'))
from train_transformer import get_model, adjust_tokenizer, compute_metrics, get_encode_labels, tokenize_function, generate_tokenized_dataset, get_datasets, test_model

def get_base_dataframes(pdf_target):
    pdf_target['matched'] = np.where(pdf_target['target'] == pdf_target['pred'], 1, 0)
    df_result = pdf_target.groupby(['target'], as_index=False)['matched'].count()
    df_result.columns = ['target', 'total_records']

    df_result['matched'] = pdf_target.groupby(['target'], as_index=False)['matched'].sum()['matched']

    df_result['percentage_matched'] = df_result['matched'] / df_result['total_records']
    return pdf_target, df_result

def draw_sanky_chart(pdf_X_test, top_OGs):
    colors_base = dict([(x, f"rgba{tuple(np.random.randint(256, size=3)) + (0.8,)}") for x in list(set(pdf_X_test['target'].unique()).union(set(pdf_X_test['pred'])))])
    labels = [x + '_True' for x in list(pdf_X_test['target'].unique())] + [x + '_Pred' for x in list(pdf_X_test['pred'].unique())]
    
    colors = [colors_base[x.replace('_True', '').replace('_Pred', '')] for x in labels]
    pdf_source_targets = pdf_X_test.groupby(['target', 'pred'], as_index=False)['matched'].count()
    
    target = {}
    sources = []
    targets = []
    values = []
    link_color = []

    for i, row in pdf_source_targets.iterrows():
        from_t = row['target'] + '_True'
        to_t = row['pred'] + '_Pred'
        from_t_index = labels.index(from_t)
        to_t_index = labels.index(to_t)
        value = row['matched']

        sources.append(from_t_index)
        targets.append(to_t_index)
        values.append(value)
        link_color.append(colors_base[row['target']])
        
    HTML("""
        <style>
            .sankey-link:hover {
                fill: gold !important;
                fill-opacity: 1 !important;
            }
        </style>
    """)
    
    # override gray link colors with 'source' colors
    opacity = 0.4
    link_color = [x.replace("0.8", str(opacity)) for x in link_color]

    fig = go.Figure(data=[go.Sankey(
        valueformat = ".0f",
        valuesuffix = "TWh",
        # Define nodes
        node = dict(
          pad = 15,
          thickness = 15,
          line = dict(color = "black", width = 0.5),
          label =  labels,
          color =  colors
        ),
        # Add links
        link = dict(
          source =  sources,
          target =  targets,
          value =  values,
          # label =  data['data'][0]['link']['label'],
          color =  link_color
    ))])

    fig.update_layout(title_text=f"Prediction Distribution [Top OGs {top_OGs}]",
                      font_size=10, height=3000)
    fig.show()

def get_highest_two(row):
    target = row['target']
    pred_classes = row['pred_classes']
    total_records_pred = np.array(row['total_records_pred'])
    total_records = row['total_records']
    
    sorted_idx = total_records_pred.argsort()
    
    if len(sorted_idx) > 2:
        top_2_GOs = [pred_classes[sorted_idx[-1]], pred_classes[sorted_idx[-2]]]
        top_2_count = [total_records_pred[sorted_idx[-1]], total_records_pred[sorted_idx[-2]]]
        top_2_perc = [total_records_pred[sorted_idx[-1]]/total_records, total_records_pred[sorted_idx[-2]]/total_records]
    else:
        top_2_GOs = [pred_classes[sorted_idx[-1]], 'None']
        top_2_count = [total_records_pred[sorted_idx[-1]], 0.0]
        top_2_perc = [total_records_pred[sorted_idx[-1]]/total_records, 0.0]
    
    return {
        'top_2_GOs': top_2_GOs,
        'top_2_count': top_2_count,
        'top_2_perc': top_2_perc
    }

def calculate_text_length(txt_field):
    return len(txt_field.replace('.', '').split())

def calculate_top_OGs(pdf_X_test, df_result, pdf_train):
    pdf_source_targets = pdf_X_test.groupby(['target', 'pred'], as_index=False)['matched'].count()
    df_pred_list = pdf_source_targets.groupby('target').agg({'pred': lambda x: list(x), 'matched': lambda x: list(x)})
    df_pred_list = df_pred_list.reset_index(level=0)
    df_pred_list.columns = ['target', 'pred_classes', 'total_records_pred']
    
    pdf_merged = pd.merge(df_pred_list, df_result, on='target', how='left')
    
    li_top_two_OGs = []
    li_top_two_counts = []
    li_top_two_per = []
    for i, row in pdf_merged.iterrows():
        val = get_highest_two(row)   
        li_top_two_OGs.append(val['top_2_GOs'])
        li_top_two_counts.append(val['top_2_count'])
        li_top_two_per.append(val['top_2_perc'])
        
    pdf_merged['top_2_GOs'] = li_top_two_OGs
    pdf_merged['top_2_count'] = li_top_two_counts
    pdf_merged['top_2_perc'] = li_top_two_per

    pdf_train = pdf_train[['TEXT_FINAL', 'target']].dropna()
    pdf_train['text_length'] = pdf_train['TEXT_FINAL'].map(calculate_text_length)
    pdf_base_count = pdf_train.groupby('target', as_index=False).agg(count_in_training_set=('TEXT_FINAL', 'count'), text_length_average=('text_length', 'mean'))
    # pdf_base_count.columns = ['target', 'count_in_training_set']

    pdf_final = pd.merge(pdf_merged, pdf_base_count, on='target', how='left')
    pdf_final['count_ratio'] = [x/5 if x/5 > 10 else 10 for x in list(pdf_final['total_records'])]
    
    return pdf_final

def calculate_performance(y_true, y_pred):
    precision_recall_fscore = precision_recall_fscore_support(y_true, y_pred, average='macro')

    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    recall_weighted = recall_score(y_true=y_true, y_pred=y_pred, average='weighted')
    precision_weighted = precision_score(y_true=y_true, y_pred=y_pred, average='weighted')
    f1_weighted = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')

    return {"accuracy": accuracy, "precision": precision_recall_fscore[0], "recall": precision_recall_fscore[1], "f1": precision_recall_fscore[2], 
            "recall_weighted": recall_weighted, "precision_weighted": precision_weighted, "f1_weighted": f1_weighted}

def get_valid_runs(exp, primary_metric='temporal_test_f1_weighted'):
    counter = 0
    all_runs = exp.get_runs(include_children=True)
    dic_runs = {}
    for i, run in enumerate(all_runs):
        # break
        metrics = run.get_metrics()
        if primary_metric in metrics:
            dic_runs[run.id] = {
                'run': run,
                'metrics': metrics
            }
        counter+=1
        
    print(f'Total runs returned: [{len(dic_runs.keys())}]')
    return dic_runs

def get_highest_performing_model(dic_runs, primary_metric, top_OGs, second_metric = "test_f1_weighted",
    third_metric = 'temporal_test_f1',
    forth_metric = 'test_f1'):

    li_test_values = []
    best_performing_run = None
    for run_id in dic_runs:
        if 'HD_e8d5cb0f-6d51-4837-bef6-be3e8072eb38' in run_id:
            continue

        run = dic_runs[run_id]['run']
        dataset = run.get_details()['inputDatasets'][0]['dataset']
        if 'top' in dataset.tags and dataset.tags['top'] != top_OGs:
            continue

        temporal_test_f1_weighted = dic_runs[run_id]['metrics'][primary_metric]
        if len(li_test_values) > 0 and temporal_test_f1_weighted > max(li_test_values):
            best_performing_run = dic_runs[run_id]

        li_test_values.append(temporal_test_f1_weighted)
    
    run = best_performing_run['run']
    for dataset in run.get_details()['inputDatasets']:
        if dataset['consumptionDetails']['inputName'] == 'train':
            train_dataset = dataset
        elif dataset['consumptionDetails']['inputName'] == 'temporal_test':
            temporal_dataset = dataset

    print(f'run id: {run.id}')
    print(f'{primary_metric}: {best_performing_run["metrics"][primary_metric]} - {second_metric}: {best_performing_run["metrics"][second_metric]}')
    print(f'{third_metric}: {best_performing_run["metrics"][third_metric]} - {forth_metric}: {best_performing_run["metrics"][forth_metric]}')
    print(f'The datasets are for logic: [{train_dataset["dataset"].tags["logic"]}] and total OGs of [{train_dataset["dataset"].tags["top"]}]')
    print(f'Train dataset name: {train_dataset["dataset"].name}, V:{train_dataset["dataset"].version}')
    print(f'Temporal test dataset name: {temporal_dataset["dataset"].name}, V:{temporal_dataset["dataset"].version}')

    return best_performing_run

def get_dataset(best_performing_run):
    run = best_performing_run['run']

    ds_train = None
    ds_temporal_test = None
    ds_val= None
    ds_test = None

    for dataset in run.get_details()['inputDatasets']:
        if dataset['consumptionDetails']['inputName'] == 'train':
            ds_train = dataset["dataset"]
        elif dataset['consumptionDetails']['inputName'] == 'validation':
            ds_val = dataset["dataset"]
        elif dataset['consumptionDetails']['inputName'] == 'test':
            ds_test = dataset["dataset"]
        elif dataset['consumptionDetails']['inputName'] == 'temporal_test':
            ds_temporal_test = dataset["dataset"]            

    # print(f'The datasets are for logic: [{ds_train.tags["logic"]}] and total OGs of [{ds_train.tags["top"]}]')
    # print(f'Train dataset name: {ds_train.name}, V:{ds_train.version}')
    # print(f'Train dataset name: {ds_temporal_test.name}, V:{ds_temporal_test.version}')

    return {
        'ds_train': ds_train,
        'ds_temporal_test': ds_temporal_test,
        'ds_val': ds_val,
        'ds_test': ds_test
    }

def get_transformer_objects(run, pdf, top_OGs, target_field='target'):
    dir = f'output_{top_OGs}'

    isdir = os.path.isdir(dir)
    if isdir:
        shutil.rmtree(dir)

    run.download_files(prefix="outputs/model", output_directory=dir, timeout_seconds=6000)
 
    model_directory = f'{dir}/outputs/model'
    print(f'the output path: [{model_directory}]')

    num_labels = len(pdf[target_field].unique())
    print(f'Number of labels: [{num_labels}]')

    model = AutoModelForSequenceClassification.from_pretrained(model_directory, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    le=joblib.load(model_directory + '/labelEncoder.joblib')
    print('Model objects and their dependencies are loaded')

    return {
        'model': model,
        'tokenizer': tokenizer,
        'le': le
    }

def predict(run, pdf_train, pdf, top_OGs, target_field='target'):
    transformers_obj = get_transformer_objects(run, pdf_train, top_OGs, target_field)
    model = transformers_obj['model']
    tokenizer = transformers_obj['tokenizer']
    le = transformers_obj['le']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    model.zero_grad()
    print(device)

    fields = ['TEXT_FINAL', 'target']
    target_name = 'target'
    text_field_name = 'TEXT_FINAL'

    pdf = pdf[fields].dropna()
    print(f'Dataframe shape: {pdf.shape}')
    ds, tokenized_ds = generate_tokenized_dataset(pdf, fields, le, target_name, text_field_name, tokenizer)

    trainer = Trainer(model=model, compute_metrics=compute_metrics, tokenizer=tokenizer)

    pred_result = trainer.predict(tokenized_ds)
    print('Prediction completed')
    pred = np.argmax(pred_result.predictions, axis=1)
    labels = pred_result.label_ids

    pdf['pred_enc'] = pred
    pdf['pred'] = le.inverse_transform(pdf['pred_enc'])

    return pdf

def plot_accuracy_count_plot(pdf, ds, top_OGs):
    fig = go.Figure(data=go.Scatter(y=pdf['count_in_training_set'],
                                    x=pdf['percentage_matched'],                                
                                    text=pdf['target'],
                                    marker=dict(
                                        size=pdf['count_ratio'],
                                        color=pdf['text_length_average'],
                                        colorbar=dict(
                                            title="Average text length<br />Training Set"
                                        ),
                                        colorscale='Viridis'
                                        ),
                                    mode='markers'))

    fig.update_yaxes(type="log")
    fig.update_layout(title_text=f"Number of Training Sample vs Prediction Accuracy in Temporal Test<br />[Top OGs: {top_OGs}] - Logic [{ds.tags['logic']}]",
                      yaxis_title="Record in Training Set",
                      xaxis_title="Accuracy",
                      font_size=15,
                      width=1500, 
                      height=1000)

    return fig
