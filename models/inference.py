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
device = 'cuda:1' if cuda.is_available() else 'cpu'

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# free_gpu_cache()         
logger = get_logger(__name__, '../logs/inference_tweet_ner_val_larger_memory.log', use_formatter=False)



def load_token_model(model_name,task,framework):
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained('pretrained/{}'.format(model_name.replace('/','_')))
    if framework.lower() == 'pytorch':
        from_tf = False
    else:
        from_tf = True
    if task == 'pos':
        task = 'token-classification'
    model = AutoModelForTokenClassification.from_pretrained('pretrained/{}'.format(model_name.replace('/','_')),from_tf=from_tf).to('cpu')
    nlp = pipeline(task.lower(), model=model, tokenizer=tokenizer,aggregation_strategy="simple",device=0)
    # int8_model = OptimizedModel.from_pretrained('Intel/distilbert-base-uncased-finetuned-conll03-english-int8-static',)
    return nlp

def load_pipeline_model(model_name,task,framework):
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained('pretrained/{}'.format(model_name.replace('/','_')))
    if framework.lower() == 'pytorch':
        from_tf = False
    else:
        from_tf = True
    
    # if task == 'sentiment':
        # if 'robertaa' in model_name.lower():
        #     from transformers import RobertaForSequenceClassification
        #     from transformers import RobertaTokenizer
        #     tokenizer = RobertaTokenizer.from_pretrained('pretrained/{}'.format(model_name.replace('/','_')))
        #     model = RobertaForSequenceClassification.from_pretrained('pretrained/{}'.format(model_name.replace('/','_')),from_tf=from_tf)
        # elif 'bert' in model_name.lower():
        #     # from transformers import BertTokenizer
        #     from transformers import BertForSequenceClassification
        #     # tokenizer = BertTokenizer.from_pretrained('pretrained/{}'.format(model_name.replace('/','_')))
        #     model = BertForSequenceClassification.from_pretrained('pretrained/{}'.format(model_name.replace('/','_')),from_tf=from_tf)
        # else:
    model = AutoModelForSequenceClassification.from_pretrained('pretrained/{}'.format(model_name.replace('/','_')),from_tf=from_tf)
    if from_tf == False:
        model = model.to(device)

    if task == 'sentiment':
        task = 'sentiment-analysis'
        nlp = pipeline(task.lower(), model=model, tokenizer=tokenizer,device=0)
    
    return nlp

def load_model(model_name,m_type,task,framework):
    if m_type == 'token':
        model = load_token_model(model_name,task,framework)
        # model = Tokenizer(model_name,task=task,framework=framework).load_model()
        return model
    elif m_type == 'flair':
        model = SequenceTagger.load('pretrained/flair/{}.pt'.format(model_name.replace('/','_'))).to(device)#Flair(model_name,'',task=task).load_model()
        return model
    elif m_type == 'pipeline':
        model = load_pipeline_model(model_name,task,framework)
        return model


def verify(length,results,m_type='token',task='ner'):
    # logger.info('len: {}'.format(length))
    # logger.info('results: {}'.format(results))
    if task == 'sentiment':
        return sentiment_tag[results[0]['label'].lower()]
    predictions = []
    if m_type == 'token':
        for item in results:
            # logger.info('result item: {}'.format(item))
            if item == []:
                predictions.append('O')
                continue
            elif isinstance(item,list):
                item = item[0]
            try:
                entity = item['entity']
            except:
                entity = item['entity_group']
            if task =='ner':
                if entity in ner_map:
                    entity = ner_map[entity]
                # if 'I-' in entity:
                #     entity = entity.replace('I-','B-')
                # elif '-' not in entity:
                #     entity = 'B-' + entity
                predictions.append(entity)
            elif task == 'pos':
                predictions.append(pos_map.get(entity,entity))
            
    elif m_type == 'flair':
        if task == 'ner':
            predictions = ['O']*length
            for entity in results.get_spans(task):
                for i in range(entity.tokens[0].idx-1, entity.tokens[-1].idx):
                    predictions[i] = 'B-{}'.format(entity.get_label(task).value)    
        elif task == 'pos':
            predictions = []
            for entity in results.get_labels():
                predictions.append(pos_map.get(entity.value,entity.value)) 
    return predictions

def write_result(evaluation_hf,inference_time,model_name,dataset_name,task,addition):
    path = '../evaluation/{}_{}_{}_results.csv'.format(dataset_name,addition,task)
    if not os.path.exists(path):
        df = pd.DataFrame(columns=['model_name','task','dataset','cost'])
    else:
        df = pd.read_csv(path,index_col=0)
    result = {'model_name':model_name,'task':task,'dataset':dataset_name,'cost':inference_time}
    
    for key,value in evaluation_hf.items():
        if isinstance(value,dict):
            for k,v in value.items():
                col_name = key+'_'+k
                result[col_name] = round(v,3)
        else:
            result[key] = round(value,3)
    df = df.append(result,ignore_index=True)
    df.to_csv(path)

# inference(model,dataset,task,save_path)
def inference(model,dataset,task,model_save_path,m_type,dataset_name):
    # model_name = 'Davlan/xlm-roberta-large-ner-hrl' #'Davlan/xlm-roberta-base-ner-hrl' #'dslim/bert-base-NER'

    # model = load_model(model_name)
    start = time.time()

    y_pred = []
    try: size = len(dataset[list(dataset[:].keys())[0]])
    except: size = len(dataset[label])
    for i in range(size):
        if dataset_name == 'tweet_eval':
            s1 = dataset['text'][i]
            if task != 'sentiment':
                s1 = s1.split()
        elif dataset_name == 'glue':
            s1 = dataset['sentence'][i]
            if task != 'sentiment':
                s1 = s1.split()
        else:
            s1 = dataset['tokens'][i]
        if i % 1000 == 0:
            logger.info('- sentence id: {}'.format(i))
        # logger.info('s1: {}'.format(s1))
        # s2 = (' ').join(s1)#.lower()
        # logger.info('s1: {}\ns2:{}'.format(s1, s2))

        # logger.info('----- predict full sentence -----')
        # results = model(s2)
        # results = verify(len(s1),results,labels)
        # logger.info('gt: {}, prediction: {}'.format(dataset[i]['ner_tags'],results))

        # logger.info('---- predict tokens ----')
        if m_type == 'token' or m_type == 'pipeline':
            results = model(s1)
        elif m_type == 'flair':
            results = Sentence(s1)
            model.predict(results)
        results = verify(len(s1),results,m_type,task)
        # gt = [num2tag[val] for val in dataset[i]['ner_tags']]
        # logger.info('prediction ({}): {}'.format(len(results),results))
        y_pred.append(results)
    end = time.time()
    inference_time = round((end-start)/size,3)
    logger.info('inference time: {}'.format(inference_time))

    # write down the prediction for the current model
    convert_label(model_save_path,y_pred,task)

    return inference_time

def convert_label(path,tag_list,task):
    if task == 'ner':
        num2tag = num2ner
    elif task == 'pos':
        num2tag = num2pos
    df = pd.DataFrame(tag_list).fillna('')
    if task != 'sentiment':
        for i in range(len(num2tag)):
            df = df.replace(i,num2tag[i])
    df.to_csv(path)

def write_gt(dataset,path,task,purpose,dataset_name):
    if not os.path.exists(path):
        # evaluate: model inference
        if purpose == 'evaluate':
            if task == 'sentiment':
                tag_list = dataset['label']
            else:
                tag_list = dataset['{}_tags'.format(task)]
            convert_label(path,tag_list,task)
        # experiment: gather gt and evaluate on SOA models
        if purpose == 'experiment':
            if task == 'ner':
                model_name = 'flair/ner-multi'
                m_type = 'flair'
            elif task == 'pos':
                model_name = 'flair/upos-english'
                m_type = 'flair'
            elif task == 'sentiment':
                model_name = 'cardiffnlp/twitter-roberta-base-sentiment'
                m_type = 'pipeline'
            model = load_model(model_name,m_type,task,'pytorch')
            inference_time = inference(model,dataset,task,path,m_type,dataset_name)


def record_evaluation(model_path,gt_path,inference_time,dataset_name,task,addition):
    y_pred = remove_nan(pd.read_csv(model_path,index_col=0).values.tolist())
    y_true = remove_nan(pd.read_csv(gt_path,index_col=0).values.tolist())
    if task == 'pos':
        y_true = convert_to_b(y_true)
        y_pred = convert_to_b(y_pred)
    if task == 'sentiment':
        from sklearn.metrics import classification_report
        evaluation = classification_report(y_true, y_pred,output_dict=True)
    else:
        from seqeval.metrics import classification_report
        evaluation = classification_report(y_true, y_pred)
        logger.info(evaluation)
        seqeval_hf = evaluate.load('seqeval')
        evaluation = seqeval_hf.compute(predictions=y_pred, references=y_true)
    logger.info(evaluation)
    
    # logger.info(evaluation_hf)
    write_result(evaluation,inference_time,model_name,dataset_name,task,addition)
    # write_result(evaluation_hf,inference_time,model_name,dataset_name,task)

if __name__ == '__main__':
    task =  'ner' #'sentiment' #'ner' #'pos'
    purpose = 'experiment' # 'experiment' 'evaluate'

    dataset_name = 'tweet_eval' #'tweet_eval'#'conll2003'
    addition = '_val' #'_val' # ''

    if dataset_name == 'conll2003':
        dataset = load_dataset(dataset_name, split='test')
        dataset_path = '../evaluation/{}'.format(dataset_name)
    elif dataset_name == 'tweet_eval':
        dataset = load_dataset(dataset_name, 'sentiment', split='test')
        dataset_path = '../evaluation/{}'.format(dataset_name+addition)
    elif dataset_name == 'glue':
        dataset = load_dataset(dataset_name, 'sst2', split='validation')
    
    if not os.path.isdir(dataset_path):
        os.mkdir(dataset_path)

    gt_save_path = '{}/{}_gt.csv'.format(dataset_path,task)

    size = len(dataset)//2
    if 'val' in addition:
        dataset = dataset[:size]
    else:
        dataset = dataset[size:len(dataset)-1]
    write_gt(dataset,gt_save_path,task,purpose,dataset_name)
    
    df_model = pd.read_csv('model_{}.csv'.format(task))
    for i, row in df_model.loc[:].iterrows():
        model_name = row['model_name']
        # pass ground truth model
        if model_name == 'flair/ner-multi' or model_name == 'flair/upos-english': continue

        m_type = row['type']
        # if m_type != 'token':
        #     continue
        model_save_path = '{}/{}_{}.csv'.format(dataset_path,task,model_name.replace('/','_'))

        logger.info('======== model_name: {}, task: {}, model_type: {} ========='.format(model_name,task,m_type))
        logger.info('model_path: {}'.format(model_save_path))

        free_gpu_cache()
        try:
            model = load_model(model_name,m_type,task,row['framework'])
        except BaseException:
            logger.exception("An exception was thrown!")
            continue
        inference_time = inference(model,dataset,task,model_save_path,m_type,dataset_name=dataset_name) # row['dataset']
        # inference_time = 0.14
        
        record_evaluation(model_save_path,gt_save_path,inference_time,dataset_name,task,addition)
    