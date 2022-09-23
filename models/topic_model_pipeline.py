import pandas as pd

from datasets import load_dataset
from flair.data import Sentence
from flair.models import SequenceTagger
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import AutoModelForSequenceClassification
from transformers import pipeline
from tokenizer_model import Tokenizer
from flair_model import Flair
import evaluate
import topic_model_evaluation
from functools import partial
import data

# import elasticsearch
from pathlib import Path
# from eland.ml.pytorch import PyTorchModel
# from eland.ml.pytorch.transformers import TransformerModel

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
import time

import os, sys
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/logger')
import logging
from logger.logger import get_logger
import torch
from scipy.special import softmax
import csv
# from numba import cuda
from util import *
from config import *
from torch import cuda
import topic_finetune
from tqdm import tqdm
import torch

logger = get_logger(__name__, '../logs/inference_tweet_topic_larger_memory.log', use_formatter=False)



def generate_train_data():
    df_=pd.DataFrame()
    corpus,label=topic_model_evaluation.load_truth(True)
    df_['labels']=label
    df_['text']=corpus
    return df_[df_['labels']!=-100]

def evaluation_func(model_name):
    return  partial(topic_model_evaluation.evaluate,model_name)

def topic_model_pipeline():
    task =  'topic' 
    
    df_model = pd.read_csv('model_{}.csv'.format(task))
    for i, row in df_model.loc[:].iterrows():
        model_name = row['model_name']
        dataset_path=row['data_url']
        m_type = row['type']
        model_type=row['framework']
        logger.info('======== model_name: {}, task: {}, model_type: {} ========='.format(model_name,task,m_type))

        free_gpu_cache()
        evaluation_func(model_name)(model_type)

    logger.info('======== model_name: {}, task: {}, model_type: {} ========='.format(model_name,'selecting',m_type))
    best_model,best_type=topic_model_evaluation.select_best_model()
    logger.info('======== model_name: {}, task: {}, model_type: {} ========='.format(best_model,'selected',best_type))
    evaluation_func(best_model)(best_type,'ground_truth')
    logger.info('======== Finetune =========')
    training_set, val_set = data.createDataset(generate_train_data())
    for i, row in df_model.loc[:].iterrows():
        model_name = row['model_name']
        dataset_path=row['data_url']
        m_type = row['type']
        model_type=row['framework']
        logger.info('======== model_name: {}, task: {}, model_type: {} ========='.format(model_name,task,m_type))
        topic_finetune.finetune(model_name,model_type, training_set, val_set, 10)
        free_gpu_cache()
    
def topic_model_inference(model_name,topic_filters=None,dataset='tweet_eval'):
    #read model metadata

    df_model = pd.read_csv('model_topic.csv')
    #read id2label mapping 
    df_label=pd.read_csv('../topic_evaluation/topic_id2label.csv')
    id2label, label2id, label_filters ={},{},{}
    for id,label in zip(df_label['id'],df_label['label']):
        label=label.lower()
        id2label[id]=label
        label2id[label]=id
    num_labels=len(id2label)
    if(topic_filters=='all'):
        id_filters=set(range(num_labels))
    else:
        topics=topic_filters.split(',')
        id_filters=[label2id[x] for x in topics]

    model_type=df_model[df_model['model_name']==model_name]['framework'].iloc[0]
    print(model_type)
    model=topic_finetune.load_model(model_name=model_name,model_type=model_type,num_labels=num_labels)
    if(dataset!='tweet_eval'):
        raise NotImplementedError
    training_set, val_set = data.createDataset(generate_train_data(),train_size=0.7)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset, val_dataset = data.getNewDataset(tokenizer,training_set,val_set)
    one_hot=False
    if(model_type=='TF'):
        val_params['batch_size']=5
        one_hot=True
    val_loader = DataLoader(val_dataset, **val_params)
    from datetime import datetime
    version = datetime.timestamp(datetime.now())
    prediction_logits=[]
    prediction_labels=[]
    with torch.no_grad():
        with tqdm(total=len(val_loader)) as pbar:
            for _, (text,labels) in enumerate(val_loader, 0):
                logits,loss=topic_finetune.foward_model(model,text,labels,num_labels,one_hot)
                prediction=np.argmax(logits,axis=1)
                size=logits.shape[0]
                for i in range(size):
                    if(prediction[i] in id_filters):
                        prediction_logits.append(logits[i])
                        prediction_labels.append(prediction[i])
                pbar.update(1)
    del model
    torch.cuda.empty_cache()
    prediction_logits=np.array(prediction_logits)
    prediction_labels=np.array(prediction_labels)
    columns=[id2label[x] for x in range(num_labels)]
    columns.extend(['label'])
    result=pd.DataFrame(np.vstack((prediction_logits.T,prediction_labels)).T,columns=columns)
    result=result.astype({'label':'int32'})
    result.to_csv('../topic_evaluation/prediction_result_{}.csv'.format(version),index=False)

                  


if __name__ == '__main__':
    
    topic_model_inference('mrm8488/bert-tiny-finetuned-yahoo_answers_topics',topic_filters='all',dataset='tweet_eval')
