# Importing stock ml libraries
import numpy as np
import pandas as pd
import transformers
import bitsandbytes as bnb
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import BertTokenizer, BertModel, BertConfig
from datasets import load_dataset, load_metric
import urllib.request

from config import *
from util import *
from data import *
from inference import *

import os, sys
import time

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/logger')
import logging
from logger.logger import get_logger
logger = get_logger(__name__, '../logs/sentiment_gt.log', use_formatter=False)

m_type = 'pipeline'
task = 'sentiment'
model_name = 'cardiffnlp/twitter-roberta-base-sentiment'
dataset_name = 'jigsaw_toxicity_pred' #'tweet_eval'#'conll2003'
addition = 'train' #'_val' # ''
dataset = load_dataset(dataset_name, data_dir='../train/Toxic-Comment-Classification-Challenge/data')
logger.info('----------')
logger.info(f'model_name: {model_name}')
logger.info('dataset_name: {}, len: {}'.format(dataset_name,len(dataset['train'])))
logger.info(dataset['train'].features)
logger.info(dataset['train'][:4])

df = pd.DataFrame()
for columns in list(dataset['train'].features.keys()):
    df[columns] = dataset['train'][columns]
    # "toxic","severe_toxic","obscene","threat","insult","identity_hate"

dir_path = '../train/{}_{}'.format(dataset_name,task)
gt_save_path = dir_path+'/gt.csv'
if not os.path.isdir(dir_path):
    os.mkdir(dir_path)

df.to_csv(gt_save_path)
logger.info('---------')
logger.info(df.head())

# download label mapping
labels=[]
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]
print(labels)


# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
 
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# inference(model,dataset,task,save_path)
def execute(model,tokenizer,dataset,m_type,dataset_name,size,batch_size=8):
    start = time.time()

    y_pred = {k:[] for k in labels}
    # try: size = len(dataset[list(dataset[:].keys())[0]])
    # except: size = len(dataset[label])
    i = 0
    while i < size:
        if i+batch_size < size:
            s1 = [preprocess(s) for s in dataset['comment_text'][i:i+batch_size]]
            # print(s1)
        else:
            s1 = [preprocess(s) for s in dataset['comment_text'][i:size]]
        i += batch_size
        encoded_input = tokenizer.batch_encode_plus(
            s1,
            # None,
            max_length=200,
            pad_to_max_length=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )
            # encoded_input = tokenizer(s1, return_tensors='pt')
            # s1 = {key: val[i] for key, val in dataset.items()}
        output = model(**encoded_input)
        # logger.info(output)
        # results = verify(len(s1),results,m_type,task)
        scores = output[0].detach().numpy() # output[0][0].detach().numpy()
        scores = softmax(scores,axis=1)
        # logger.info(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        # logger.info(ranking)
        if scores.shape[0] != batch_size:
            print(scores.shape[0])
        for item in range(scores.shape[0]):
            score = scores[item]
            rank = ranking[item]
            for j in range(scores.shape[1]):
                l = labels[rank[j]]
                s = score[rank[j]]
                y_pred[l].append(s)
            # print(f"{j+1}) {l} {np.round(float(s), 4)}")

        if i % 2000 == 0:
            logger.info('- sentence id: {}'.format(i))
            logger.info('ranking: {}'.format(ranking))
            df_tmp = df.iloc[:i]
            for key in list(y_pred.keys()):
                df_tmp[key] = y_pred[key]
            df_tmp.to_csv(dir_path+'/gt_tmp.csv')

        
    end = time.time()
    inference_time = round((end-start)/size,3)
    logger.info('inference time: {}'.format(inference_time))

    # write down the prediction for the current model
    # convert_label(model_save_path,y_pred,task)
    # df = pd.read_csv('../train/Toxic-Comment-Classification-Challenge/data/train.csv')
    
    return y_pred

tokenizer = load_tokenizer(model_name)
# encoded_input = tokenizer([preprocess(s1) for s1 in list(dataset['train']['comment_text'])], truncation=True,padding=True,return_tensors='pt')
# print(encoded_input.keys)
model_path = './pretrained/{}'.format(model_name.replace('/','_'))
model = AutoModelForSequenceClassification.from_pretrained(model_path)
# task = 'sentiment-analysis'
# nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer,device=0)
y_pred = execute(model,tokenizer,dataset['train'],m_type,dataset_name,size=len(dataset['train']['comment_text']))


for key in list(y_pred.keys()):
    df[key] = y_pred[key]
    
df.to_csv(gt_save_path)



