# Importing stock ml libraries
import gc
import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
from seqeval.metrics import f1_score
import evaluate
# metric = evaluate.load('seqeval')

from config import *
from util import *
from data import *
from inference import *
from new_finetune import *

import os, sys
import time
import glob

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/logger')
import logging
from logger.logger import get_logger
logger = get_logger(__name__, '../logs/evaluate.log', use_formatter=False)

dataset_name = 'tweet_eval' #'tweet_eval'#'conll2003'
addition = 'train' #'_val' # ''
logger.info('dataset_name: {}'.format(dataset_name))

tasks = ['ner','pos','sentiment']
model_name_list = ['model_{}.csv'.format(task) for task in tasks]
model_list = [list(pd.read_csv(model_name,index_col=0)['model_name']) for model_name in model_name_list]

model_names = glob.glob('./finetuned/*')


def eval(model_name,task,label_list):
    val_loader = DataLoader(val_dataset, **val_params)
    model_path = "./finetuned/{}-finetuned-{}.pt".format(model_name.replace('/','-'),dataset_name)
    
    torch.cuda.empty_cache()
    # To get the results on the validation set. This data is not seen by the model
    model = FinetuneClass(model_name, task,num_labels=len(label_list[task]))
    model.load_state_dict(torch.load(model_path),strict=False)
    model.to(device)
    valid(model, val_loader,task) 

for model_name in model_names:
    logger.info('model_name: {}'.format(model_name))
    user_name = model_name.split('-')[0]
    rest = '-'.join(model_name.split('-')[1:-1])
    model_name = f'{user_name}/{rest}'
    task = tasks[[1 if model_name in l else 0 for l in model_list ].index(1)]
    logger.info('task: {}'.format(task))

      