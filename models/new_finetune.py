# Importing stock ml libraries
import gc
import glob
import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import bitsandbytes as bnb
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from flair.data import Sentence
from flair.models import SequenceTagger
from datasets import load_dataset, load_metric
from seqeval.metrics import f1_score
import evaluate
# metric = evaluate.load('seqeval')

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
logger = get_logger(__name__, '../logs/evaluate_ner.log', use_formatter=False)



def pytorch_trainer(train_dataset,model_name,dataset_name,task,purpose):
    
    logger.info('-----')
    logger.info("TRAIN tokenized dataset length: {}".format(len(train_dataset)))
    logger.info("TRAIN tokenized dataset labels: {}".format(train_dataset.labels[0]))
    
    training_loader = DataLoader(train_dataset, **train_params)
    
    model_path = "./finetuned/{}-finetuned-{}.pt".format(model_name.replace('/','-'),dataset_name)
    if purpose == 'experiment':
    # if not os.path.exists(model_path):
        # model
        try:
            model = FinetuneClass(model_name, task,num_labels=len(label_list[task])).to(device)
        except Exception as e:
            logger.exception("Exception Occured while loading the model: "+ str(e))
        # optimizer = torch.optim.AdamW(params =  model.parameters(), lr=LEARNING_RATE)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        # optimizer = bnb.optim.Adam8bit(model.parameters(), lr=0.001, betas=(0.9, 0.995)) # add bnb optimizer
        # optimizer = bnb.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.995), optim_bits=8) # equivalent
        
        for epoch in range(EPOCHS):
            train(model,training_loader,epoch,optimizer)
            torch.save(model.state_dict(), model_path)




def train(model,training_loader,epoch,optimizer):
    # del variable #delete unnecessary variables 
    # gc.collect()

    scaler = torch.cuda.amp.GradScaler()
    model.train()
    for _,data in enumerate(training_loader, 0):
        # ids = data['input_ids'].to(device, dtype = torch.int)
        # mask = data['mask'].to(device, dtype = torch.int)
        # targets = data['labels'].to(device, dtype = torch.long)
        # if _ == 0:
        #     logger.info(list(data.keys()))
        input_ids = data['input_ids'].to(device)
        # token_type_ids = data['token_type_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)

        # outputs = model(ids, mask, token_type_ids)

        with torch.cuda.amp.autocast():
            output = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)[:2]
            # output = model(ids, mask, labels = targets)
        # output = model(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, labels = labels)[:2]
        loss, logits = output[:2]

        if _%5000==0:
            # logits = logits.detach().cpu().numpy()
            # label_ids = targets.to('cpu').numpy()
            logger.info(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        # scaler.scale(loss).backward()
        optimizer.step()
        # scaler.step(optimizer)
        # scaler.update()

# pytorch validation
def flat_accuracy(preds, labels):
    flat_preds = np.argmax(preds, axis=2).flatten()
    flat_labels = labels.flatten()
    return np.sum(flat_preds == flat_labels)/len(flat_labels)

def valid(model, val_dataset,task,finetune):

    val_loader = DataLoader(val_dataset, **val_params)
    torch.cuda.empty_cache()

    model.eval()
    eval_loss = 0; eval_accuracy = 0; inference_time = 0
    n_correct = 0; n_wrong = 0; total = 0
    predictions , true_labels = [],[]
    nb_eval_steps, nb_eval_examples = 0, 0
    results_dict = {}
    with torch.no_grad():
        for _, data in enumerate(val_loader, 0):
            start = time.time()

            # ids = data['input_ids'].to(device, dtype = torch.int)
            # mask = data['mask'].to(device, dtype = torch.int)
            # labels = data['labels'].to(device, dtype = torch.long)

            input_ids = data['input_ids'].to(device)
            # token_type_ids = data['token_type_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)

            # output = model(ids, mask, labels=labels)
            output = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
            loss, logits = output[:2]
            logits = logits.detach().cpu().numpy()
            # logger.info('logits size: {}'.format(np.array(logits).shape))
            # logger.info('logits: {}'.format(logits[0]))
            label_ids = labels.to('cpu').numpy()
            # logger.info('label size: {}'.format(np.array(label_ids).shape))
            # logger.info('label: {}'.format(label_ids[0]))

            # predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            predictions.extend(logits)
            true_labels.extend(label_ids)

            # accuracy = flat_accuracy(logits, label_ids)
            # accuracy = flat_accuracy(predictions,true_labels)
            eval_loss += loss.mean().item()
            inference_time += time.time() - start
            # eval_accuracy += accuracy
            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

            if _ % 5000==0:
                logger.info('loss: {}'.format(loss))
                logger.info("Validation loss: {}".format(eval_loss))
                # logger.info("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
        # print(np.array(true_labels).shape,true_labels[0].shape,type(true_labels[0]))
        # results = compute_metrics((predictions,true_labels),task)
        eval_loss = eval_loss/nb_eval_steps
        eval_cost = inference_time/nb_eval_steps
        logger.info("Validation loss: {}".format(eval_loss))
        logger.info("Validation cost: {}".format(eval_cost))
        # logger.info("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
        # logger.info('predictions: {}'.format(predictions[:2]))
        # logger.info('labels: {}'.format(true_labels[:2]))

        # logger.info('pred_tags shape: {}'.format(np.array(predictions).shape))
        # logger.info('valid_tags shape: {}'.format(np.array(true_labels).shape))

        try:
            # pred_tags = [label_list[p_i] for p in predictions for p_i in p]
            # valid_tags = [label_list[l_ii] if l_ii > 0 else label_list[0] for l in true_labels for l_i in l for l_ii in l_i]

            # pred_tags = [[label_list[task][p_i] for p_i in p] for p in predictions ]
            # valid_tags = [[label_list[task][l_ii] if l_ii > 0 else label_list[task][0] for l_ii in l_i] for l in true_labels for l_i in l ]

            if not finetune: task += '_ori'
            results = compute_metrics((predictions,true_labels),task)
            # logger.info("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
            logger.info('results: {}'.format(results))
        except Exception as e:
            # logger.info('l_ii: {}'.format(l_ii))
            logger.exception("Exception Occured while computing f1: "+ str(e))
        
        return results, eval_cost

if __name__ == '__main__':

    task =  'ner' #'sentiment' #'ner' #'pos'
    purpose = 'evaluate' # 'experiment' 'evaluate'
    finetune = False

    dataset_name = 'tweet_eval' #'tweet_eval'#'conll2003'
    addition = 'train' #'_val' # ''

    data = get_ground_truth(dataset_name,task)
    training_set, val_set = createDataset(data)

    torch.cuda.empty_cache()

    finetuned_model_list = glob.glob('./finetuned/*')
    logger.info(finetuned_model_list[:4])

    model_df = pd.read_csv('model_{}.csv'.format(task),index_col=0)

    results_dict = {}
    for i, row in model_df.iterrows():
        model_name = row['model_name']
        if i < 0: continue
        if 'flair' in model_name: continue
        tmp = model_name.replace('/','-')
        # if not f'./finetuned/{tmp}-finetuned-{dataset_name}.pt' in finetuned_model_list:
        #     continue
        
        logger.info('=========')
        logger.info('id: {}, model_name: {}'.format(id,model_name))

        try:
            tokenizer = load_tokenizer(model_name)
        except Exception as e:
            logger.exception("Exception Occured while loading the data, tokenizer: "+ str(e))
            continue
        
        try:
            train_dataset, val_dataset = getTokenizedDataset(tokenizer,training_set,val_set)
        except Exception as e:
            logger.exception("Exception Occured while tokenizing data: "+ str(e))
            continue


        print_gpu_utilization(logger)

        # pytorch training
        if finetune:
            try:
                results, cost = pytorch_trainer(train_dataset,model_name,dataset_name,task,purpose)
            except Exception as e:
                logger.exception("Exception Occured while finetuning the model: "+ str(e))
                continue
        
        # To get the results on the validation set. This data is not seen by the model
        if finetune:
            model = FinetuneClass(model_name, task,num_labels=len(label_list[task]))
            model.load_state_dict(torch.load(model_path),strict=False)
        else:
            if row['framework'].lower() == 'pytorch':
                from_tf = False
            else:
                from_tf = True
            if row['type'] == 'token':
                model = AutoModelForTokenClassification.from_pretrained('pretrained/{}'.format(model_name.replace('/','_')),from_tf=from_tf).to('cpu')
        model.to(device)

        # model evaluation
        results, cost = valid(model, val_dataset,task,finetune)

        tmp = model_name.replace('/','-')
        model_name = f'{tmp}-finetuned-{dataset_name}'
        results_dict[model_name] = {}
        for key, val in results.items():
            if isinstance(val, dict):
                for k, v in val.items():
                    results_dict[model_name][f'{key}_{k}'] = v
            else:
                results_dict[model_name][key] = val
        results_dict[model_name]['cost'] = cost
    
    df = pd.DataFrame.from_dict(results_dict,orient='index')
    eval_path = f'../evaluation/{dataset_name}_val_{task}_results.csv'
    df_ori = pd.read_csv(eval_path,index_col=0)
    # columns = model_name,task,dataset,cost,LOC_f1,LOC_number,LOC_precision,LOC_recall,MISC_f1,MISC_number,MISC_precision,MISC_recall,ORG_f1,ORG_number,ORG_precision,ORG_recall,PER_f1,PER_number,PER_precision,PER_recall,overall_accuracy,overall_f1,overall_precision,overall_recall,DATE_f1,DATE_number,DATE_precision,DATE_recall
    df['dataset'] = dataset_name
    df['task'] = task
    df['model_name'] = list(df.index)
    df.index = range(len(df))
    print(df.head())
    eval_path = f'../evaluation/{dataset_name}_val_{task}_results_.csv'
    df.to_csv(eval_path)

    # final = pd.concat([df_ori,df])
    # eval_path = f'../evaluation/{dataset_name}_val_{task}_results_new.csv'
    # final.to_csv(eval_path)
