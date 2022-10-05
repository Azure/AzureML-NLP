# Databricks notebook source
# MAGIC %md
# MAGIC # ECD: Long desc model
# MAGIC ### Predicts owner group from Long Description text input only.
# MAGIC 
# MAGIC **Randomly sampled test set is not reflective of real world performance, due to label drift.**
# MAGIC 
# MAGIC Simply guessing the most common group gives around 10% accuracy on a random split; but only 3% on the time-based split. The model delivers 73% accuracy on random test set; 42% with time-based split.
# MAGIC 
# MAGIC Recent tickets rarely have a long description in our training data, so I expect that it would perform worse in production.
# MAGIC 
# MAGIC 
# MAGIC Possible improvements:
# MAGIC * address label drift?
# MAGIC * ~~BERT model~~ (didn't help, probably due to messy text and large amount of out-of-vocabulary words)
# MAGIC * pre-processing (spell correct, ~~remove stop words~~ (doesn't help), translation (costly/slow - see helsinki_clean notebook))
# MAGIC * address class imbalance
# MAGIC * ~~"top 5" (ranking) model~~ (done)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and functions

# COMMAND ----------

# MAGIC %sh sudo apt-get install -y graphviz

# COMMAND ----------

# MAGIC %pip install tensorflow -q
# MAGIC %pip install gensim -q
# MAGIC %pip install azureml-sdk -q
# MAGIC %pip install pydot -q
# MAGIC %pip install graphviz -q
# MAGIC %pip install nltk -q

# COMMAND ----------

import os
import tensorflow
import re
import tensorflow.keras as k
import gensim
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec
from pyspark.sql.types import IntegerType 
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import string
import unicodedata
import time
import sys
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import gensim.downloader as api
from sklearn.feature_extraction.text import TfidfVectorizer
import collections
import pickle
import gzip
from os.path import exists
import pandas as pd
import gc
import time
from datetime import date
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import models
import azureml.core
from azureml.core import Workspace, Experiment, Run
from azureml.core.model import Model as AzureModel
from azureml.core.datastore import Datastore
from azureml.core.authentication import ServicePrincipalAuthentication
import datetime
import os
import tempfile
import datetime

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
english_stopwords = stopwords.words('english')

kv_name = "ScScCSV-DSAI-Lab-dev-kv"

def getCorpus():
#   corpus = api.load('text8')
#   model = Word2Vec(corpus)
#   return corpus, model
    model = api.load('glove-wiki-gigaword-100') # Best performing
#     model = api.load('fasttext-wiki-news-subwords-300')
    return None, model

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')

def cleanString(s, decontract=False, deaccent=False, depunctuate=False,
                lowercase=False, remove_stopwords=False, lemmatize=False, remove_numbers=False):
    result = s
    result = result.replace(u'\ufffd', " ") # Switch unicode replacement character for space
    result = result.replace('\xa0', ' ') # A weird string common in the input data
    if lowercase:
        result = result.lower()
    if decontract:
        result = result.replace("’","'")
        result = result.lower() # Tokenizer does this already, but this is necessary for decontraction
        result = decontracted(result)
    if depunctuate:
        translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) # Tokenizer does this already?
        result = result.translate(translator) # Tokenizer does this already?
    if deaccent:
        result = strip_accents(result)
    if remove_stopwords:
        tokens = word_tokenize(result)
        tokens = [t for t in tokens if t not in english_stopwords]
    if lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        result = ' '.join(tokens)
    if remove_numbers:
        result = re.sub(r"\b\d+\b", '', result).strip()
    return result

def plot(history):
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    #   for i,j in zip(epochs, val_acc):
    #     plt.annotate("%.4f" % j,xy=(i,j))
    plt.legend()

    plt.figure()
    plt.show()

def getMatrix(words, model, maxwords=10000, dims=100):
    matrix = np.zeros((maxwords, dims))

    random_init_words = []
    zeros_words = []

    for word, i in words:
        if i < maxwords:
            try:
                vector = model[word]
                matrix[i] = vector
            except KeyError:
                random_init_words.append(word)
                matrix[i] = np.random.rand(dims)
        else:
            zeros_words.append(word)

    return matrix, random_init_words, zeros_words


# Format of "date_split" should be a date string, e.g. "2018-11-25"
def prep(df, corpusModel, maxwords=30000, max_len=200, min_per_group=0, test_size=0.2, date_split=None):

    dims = len(corpusModel['word'])

    # Perform train/test split
    if date_split is None:
        # Random split
        X = list(df['LONG_DESC_TEXT'].values)
        y = list(df['ASSIGNED_OWNER_GROUP'].values)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    else:
        # Split by date
        X_train = list(df.query(f'REPORT_DATE < "{date_split}"')['LONG_DESC_TEXT'].values)
        y_train = list(df.query(f'REPORT_DATE < "{date_split}"')['ASSIGNED_OWNER_GROUP'].values)
        X_test = list(df.query(f'REPORT_DATE >= "{date_split}"')['LONG_DESC_TEXT'].values)
        y_test = list(df.query(f'REPORT_DATE >= "{date_split}"')['ASSIGNED_OWNER_GROUP'].values)

    # Counter handles missing values (unlike LabelEncoder, seen later on in this function...)
    groupCount = collections.Counter(y_train)

    # Note this is using the global variable "groupCounts" (a dict) rather than the groupCount spark object above
    if min_per_group > 0:
        z = [(i, j) if groupCounts[j] >= min_per_group else (i, 'Other') for i, j in zip(X_train, y_train)]
        X_train = [a[0] for a in z]
        y_train = [a[1] for a in z]
        z = [(i, j) if groupCounts[j] >= min_per_group else (i, 'Other') for i, j in zip(X_test, y_test)]
        X_test = [a[0] for a in z]
        y_test = [a[1] for a in z]
        
    t = Tokenizer(num_words=maxwords, oov_token='OOV')
    t.fit_on_texts(X_train)

    vocab_size = len(t.word_index) + 1
    wordIndex = t.word_index
    words = wordIndex.items()
    print(vocab_size, "vocabulary size")
    
    maxwords = min(vocab_size, maxwords)
    
    encoded_docs_train = t.texts_to_sequences(X_train)
    encoded_docs_test = t.texts_to_sequences(X_test)
    encoded_docs_train = pad_sequences(encoded_docs_train, maxlen=max_len, padding='post')
    encoded_docs_test = pad_sequences(encoded_docs_test, maxlen=max_len, padding='post')

    matrix, random_init_words, zeros_words = getMatrix(words, corpusModel, maxwords, dims)

    le = preprocessing.LabelEncoder()
    
    # Since label encoder can't handle missing values, a shortcut is to "peek" into our test set
    # This doesn't compromise the validity of the test results AFAIK.
    y_enc = to_categorical(le.fit_transform(y_train + y_test))
    y_train_enc = y_enc[:len(y_train)]
    y_test_enc = y_enc[len(y_train):]

    return {
          'matrix':matrix, 
          'random_init_words': random_init_words,
          'zeros_words': zeros_words,
          'tokenizer': t,
          'labelEncoder': le,
          'groupCount':groupCount,
          'X_train': encoded_docs_train, 'X_test':encoded_docs_test,
          'y_train':y_train_enc, 'y_test':y_test_enc,
         }


def store(classifier, path):
    if classifier is None or path is None or exists(path):
        return

    try:
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            models.save_model(m, fd.name, overwrite=True)
            serialized = fd.read()

        classifier['model'] = serialized
        
        pickled = pickle.dumps(classifier, protocol=4)
        with gzip.open(path, 'wb') as f:
            f.write(pickled)
    except Exception as e:
        print(repr(e))


# Load a classifier from disk
def load(path):
    if path is None:
        print("No path")
        return

    if not exists(path):
        print("Path no exists")
        return

    try:
        with gzip.open(path, 'rb') as f:
            content = f.read()
            classifier = pickle.loads(content)
            
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
                fd.write(classifier['model'])
                fd.flush()

                unserialized = models.load_model(fd.name)
            
            classifier['model'] = unserialized
    except Exception as e:
        print(repr(e))
        return 

    return classifier

# COMMAND ----------

MODELPATH = '/dbfs/mnt/ecd/models/'
if not os.path.exists(MODELPATH):
    os.makedirs(MODELPATH)

tensorflow.__version__

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the train and test data
# MAGIC Prepared prior to Microsoft Horizon engagement by DSAI as CSV. Preprocessing included HTML stripping and removal of PII using cognitive services and Presidio.

# COMMAND ----------

df = pd.read_csv('/dbfs/FileStore/ecd/ecd_tickets_cleaned_2.csv.gz')

# Convert the dates back to correct Pandas format
df['REPORT_DATE'] = pd.to_datetime(df['REPORT_DATE'])

# Filter to tickets from 2018+
df = df.query('REPORT_DATE >= "2018-04-01"')

# If we want to clean the long description strings further, do so here.
clean_texts = [cleanString(s, decontract=False, deaccent=False, depunctuate=True, lowercase=True, remove_stopwords=False, lemmatize=False, remove_numbers=False)
               for s in df['LONG_DESC_TEXT'].values]

df['LONG_DESC_TEXT'] = clean_texts

# COMMAND ----------

df.EXTERNAL_SYSTEM.value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data distributions
# MAGIC Not shown here: French text is often mixed into the English long description text.

# COMMAND ----------

# Distribution of tickets by quarter
def qdist(all_tickets):
    all_tickets['quarter'] = all_tickets['REPORT_DATE'].dt.year.astype(str) + '_' + all_tickets['REPORT_DATE'].dt.quarter.astype(str)
    dfs3 = pd.DataFrame(all_tickets.quarter.value_counts()).reset_index()
    dfs3.columns = ['quarter', 'num records']
    return dfs3.sort_values(['quarter']).plot.bar(x='quarter')

# Note that the jagged distribution is an artifact of the join to Long Descriptions.
# In reality, the distribution of tickets over quarters is fairly uniform.
qdist(df)

# COMMAND ----------

# How long are the "Long descriptions"?
# Judging by these results we should set truncate at 300 words.
lens = [len(d.split()) for d in df['LONG_DESC_TEXT']]
pd.Series(lens).hist()

# COMMAND ----------

# Label distributions over time?
group_counts = dict(df.query('REPORT_DATE >= "2021-04-01"')['ASSIGNED_OWNER_GROUP'].value_counts()[:50])
df_only_labels = df[['REPORT_DATE', 'ASSIGNED_OWNER_GROUP']]
df_only_labels['quarter'] = df_only_labels['REPORT_DATE'].dt.year.astype(str) + '_' + df_only_labels['REPORT_DATE'].dt.quarter.astype(str)
df_only_labels.loc[~(df_only_labels['ASSIGNED_OWNER_GROUP'].isin(list(group_counts.keys()))), 'ASSIGNED_OWNER_GROUP'] = 'Other'

groups_over_quarters = df_only_labels.groupby(['ASSIGNED_OWNER_GROUP', 'quarter'])[['REPORT_DATE']].count().reset_index()
groups_over_quarters.columns = ['owner', 'quarter', 'count']

groups_over_quarters = groups_over_quarters.sort_values(['quarter', 'owner'])

# Note that ITS312 and ITS322 - common in 2016 - are completely absent in 2021.
display(groups_over_quarters)

# COMMAND ----------

# Print a few examples
# Spaces are irrelevant (removed in Tokenizer)
print(list(df.sample(5)['LONG_DESC_TEXT'].values))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Split, tokenize, embed

# COMMAND ----------

# (Optionally) override given/default hyperparameters
sample = 1
epochs = 10
val_size = 0.1
test_size = 0.1
max_len = 300
maxwords = 30000

# Set date_split to None if random split is desired
date_split = '2020-01-01'

# COMMAND ----------

corpus, corpusModel = getCorpus()

data = prep(df, corpusModel, maxwords=maxwords, max_len=max_len, test_size=test_size, date_split=date_split, min_per_group=0)

gc.collect()

# COMMAND ----------

print(len(data['X_train']), 'training records;', len(data['X_test']), 'test records.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train model

# COMMAND ----------

text_input = Input(shape=(max_len,), dtype='int32', name='ticketDescription')

e = layers.Embedding(input_dim=data['matrix'].shape[0], output_dim=data['matrix'].shape[1], weights=[data['matrix']], trainable=True)(text_input)

e = layers.Dropout(0.25)(e)
c = layers.Conv1D(1024, 5, activation='relu')(e)
c = layers.MaxPooling1D(24)(c)
c = layers.Conv1D(512, 5, activation='relu')(e)
c = layers.MaxPooling1D(24)(c)
c = layers.Flatten()(c)
c = layers.Dropout(0.25)(c)

m = layers.Dense(512, activation='relu')(c)
m = layers.Dropout(0.25)(m)
m = layers.Dense(len(data['y_train'][0]), activation='softmax')(m)

m = Model(text_input, m)

from tensorflow.keras.optimizers import Adam
m.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print(m.summary())

# COMMAND ----------

import datetime
import joblib

#Store the weights temporarly, bug/issue in FUSE preventing from saving to DBFS directly
storedWeightsPath = os.path.join("/tmp/", 'ecd-{}.hdf5'.format(date.today().strftime('%Y-%m-%d')))
os.environ['WPATH'] = storedWeightsPath

# run_log_dir = experiment_log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
history = m.fit(data['X_train'],
                data['y_train'],
                validation_split=val_size,
                epochs=epochs,
                batch_size=4096,
                callbacks=[
#                   callbacks.TensorBoard(log_dir=run_log_dir, histogram_freq=1),
                  callbacks.EarlyStopping(monitor='val_categorical_accuracy', mode='max', verbose=1, patience=2),
                  callbacks.ModelCheckpoint( 
                      filepath=storedWeightsPath,
                      save_weights_only=True,
                      monitor='val_categorical_accuracy',
                      mode='max',
                      save_best_only=True),
                ]
               )
plot(history)
m.load_weights(storedWeightsPath)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Baseline guess accuracy?

# COMMAND ----------

group_counts = data['y_train'].sum(axis=0)
most_common_label_idx = np.argmax(group_counts)
most_common_label_in_train = data['labelEncoder'].inverse_transform([most_common_label_idx])[0]
print(most_common_label_in_train)

test_label_idxs = [np.argmax(c) for c in data['y_test']]
print(f'{sum(label_idx == most_common_label_idx for label_idx in test_label_idxs) * 100 / len(test_label_idxs):.2f}% baseline accuracy (always guess most common group from training)')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate on test data

# COMMAND ----------

result = m.evaluate(data['X_test'], data['y_test'], verbose=0)
testDict = dict(zip(m.metrics_names, result))
print("Overall:", testDict['categorical_accuracy'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test on free text input
# MAGIC So, it appears the model is not complete nonsense. We can see how it behaves below.

# COMMAND ----------

ticket_desc = """
gc wifi
"""

t = data['tokenizer']

preds = m.predict(
    pad_sequences(
        t.texts_to_sequences([ticket_desc]),
        maxlen=max_len, padding='post'
   )
)

i, = np.where(preds[0] == max(preds[0]))
result = data['labelEncoder'].inverse_transform([i[0]])[
    0], max(preds[0])
print(result[0], result[1])


# COMMAND ----------

# MAGIC %md
# MAGIC ## Register model
# MAGIC **TODO** Fix the connection to the key vault (and set the secrets...)

# COMMAND ----------

# Our account details
workspace_name = "ScScCPS-DSAI-AIDE-dev-mlw"
resource_group = "ScSc-DSAI-AIDE-dev-rg"
kv_name = "ScScCSV-DSAI-AIDE-dev-kv2"
# managed_identity = "ScDc-SSC-DSAI-spn"

sp_pw = dbutils.secrets.get(scope=kv_name, key="service-principal-password")
tenant_id = dbutils.secrets.get(scope=kv_name, key="tenant-id")
sp_id = dbutils.secrets.get(scope=kv_name, key="service-principal-id")
subscription_id = dbutils.secrets.get(scope=kv_name, key="subscription-id")

# Can't get Service Principal Auth to work!
svc_pr = ServicePrincipalAuthentication(tenant_id=tenant_id, service_principal_id=sp_id, service_principal_password=sp_pw)

from azureml.core.authentication import InteractiveLoginAuthentication

# interactive_auth = InteractiveLoginAuthentication(tenant_id=tenant_id)

ws = Workspace(
        workspace_name = workspace_name,
        subscription_id = subscription_id,
        resource_group = resource_group, 
#         auth = interactive_auth,
        auth = svc_pr
    )

# COMMAND ----------

timestamp = int(time.time())
fullpath = MODELPATH + 'model-{}-{}.bin.gz'.format(timestamp, testDict['categorical_accuracy'])

recommender = {
  'model': m,
  'tokenizer': data['tokenizer'],
  'le': data['labelEncoder'],
}

# COMMAND ----------

timestamp = int(time.time())

model_name = "ECD_longdesc_beta"

exp = Experiment(ws, model_name)
run = exp.start_logging(outputs=MODELPATH, snapshot_directory=None)
run.log('Training Start Date', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

acc = testDict['categorical_accuracy']

run.log('Accuracy', testDict['categorical_accuracy'])
run.log('History', history)

store(recommender, fullpath)
run.upload_file('outputs/{}'.format(fullpath), fullpath)
run.register_model(
  model_name=model_name,
  model_path='outputs/{}'.format(fullpath),
  properties = {'sampleSize':sample, 'accuracy':testDict['categorical_accuracy'], 'timestamp':timestamp},
  description="Predict owner group from ticket long description (HTML tags and punctuation removed)"
)

run.tag('timestamp', timestamp)
run.tag('accuracy', testDict['categorical_accuracy'])

run.complete()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate class imbalance

# COMMAND ----------

# TODO: Move this up and implement more loops, to make it less tedious to change features
feature_names = ['X_test']

groupCount = len(data['y_test'][0])
labelIndices = [list() for _ in range(groupCount)]
for recordIndex, record in enumerate(data['y_test']):
    # Find label
    labelIndex = np.argmax(record)
    # Add index of record to its appropriate label bin
    labelIndices[labelIndex].append(recordIndex)
    
print('Minimum records per label in test set:', min(len(labelIndex) for labelIndex in labelIndices))
print('Maximum records per label in test set:', max(len(labelIndex) for labelIndex in labelIndices))

dataPerLabel = [{feature_name: [] for feature_name in feature_names} for _ in range(groupCount)]
for i, recordIndices in enumerate(labelIndices):
    for feature_name in feature_names:
        dataPerLabel[i][feature_name] = data[feature_name][recordIndices]
    dataPerLabel[i]['y_test'] = data['y_test'][recordIndices]
    
numRecordsPerLabel = [len(labelIndex) for labelIndex in labelIndices]
accuracyPerLabel = []

for i, labelGroup in enumerate(dataPerLabel):
    if numRecordsPerLabel[i] > 0:
        result = m.evaluate([labelGroup[feature_name] for feature_name in feature_names], labelGroup['y_test'])
        testDict = dict(zip(m.metrics_names, result))
        accuracyPerLabel.append(testDict['categorical_accuracy'])
    else:
        accuracyPerLabel.append(None)
        
# # Just how many groups are there which have < min_group_size records?
# numTinyGroups = sum(1 for i in numRecordsPerLabel if i <= min_per_group)
# print('Groups with <', min_per_group, 'records assigned:', numTinyGroups)
# print('Proportion of groups having <', min_per_group, 'records:', numTinyGroups / len(numRecordsPerLabel))

# # Total number of records in the tiny groups:
# numTinyRecords = sum(i for i in numRecordsPerLabel if i <= 10)
# print('Records in groups with <', min_per_group, 'records assigned:', numTinyRecords)
# print('Proportion of total records in excluded groups:', numTinyRecords / len(data['C_test']))
        
plt.plot(numRecordsPerLabel, accuracyPerLabel, 'bo', alpha=0.2)
plt.xscale('log')
plt.xlabel('Records in group')
plt.ylabel('Accuracy')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Top X analysis
# MAGIC Was the correct group the model's first prediction? Second? Third? Fourth? Fifth?
# MAGIC 
# MAGIC Since we plan to return top 5, lets look for "top X accuracy", defined as having the correct group within the top X returned groups.

# COMMAND ----------

predictions = m.predict(data['X_test'], verbose=0)

top_x = 5

ranked_preds = []

# Each prediction in test set:
for i, preds in enumerate(predictions):
    # Get indices of top predictions
    top_pred_idx = np.argpartition(preds, -top_x)[-top_x:]
    top_vals = preds[top_pred_idx]
    # Get index of correct prediction
    true_idx = np.argmax(data['y_test'][i])
    top_pred_idx = [a[0] for a in sorted([z for z in zip(top_pred_idx, top_vals)], key=lambda x:x[1], reverse=True)]
    is_label = [pred_idx == true_idx for pred_idx in top_pred_idx]
    ranked_preds.append(is_label)

pd.DataFrame(ranked_preds)

# COMMAND ----------

rp = np.array([np.array([r for r in ranked_preds])])
print(rp.shape)
rp.sum(axis=1)

# COMMAND ----------

p_top_x = (rp.sum(axis=1) / rp.shape[1]).flatten()
p_top_x_cumulative = [np.sum(p_top_x[:i]) for i in range(1, top_x+1)]

to_graph = pd.Series(np.append(p_top_x_cumulative, 1 - p_top_x_cumulative[-1]))
to_graph.index = [f'Top {i}' for i in range(1, top_x+1)] + [f'Not in top {top_x}']

ax = to_graph.plot.bar(title='Top X accuracy (correct label within top X predictions)', figsize=(7,5), ylim=(0, 1))
ax.bar_label(ax.containers[0])

# COMMAND ----------

to_graph = pd.Series(np.append(p_top_x, 1 - p_top_x_cumulative[-1]))
to_graph.index = [f'Prediction {i}' for i in range(1, top_x+1)] + [f'Not in top {top_x}']
ax = to_graph.plot.bar(title='Proportion correct label is 1st, 2nd.. Xth prediction', figsize=(7,5), ylim=(0, 1))
ax.bar_label(ax.containers[0])

# COMMAND ----------

p_top_x_cumulative[-1]

# COMMAND ----------


