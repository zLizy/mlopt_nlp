from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import expit
import json
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import os
import data
from tqdm import tqdm
from config import *
from util import *
from torch import cuda
import torch

import logging
from logger.logger import get_logger


device = 'cuda' if cuda.is_available() else 'cpu'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/logger')

logger = get_logger(__name__, '../logs/inference_tweet_topic_larger_memory.log', use_formatter=False)



def pytorch_trainer(train_dataset,model_name,num_labels,model_type, one_hot=False):
    
    logger.info('-----')
    logger.info("TRAIN tokenized dataset length: {}".format(len(train_dataset)))
    
    training_loader = DataLoader(train_dataset, **train_params)
    
    model_path = "../topic_evaluation/{}-finetuned.pt".format(model_name.replace('/','-'))
    
    model = FinetuneClassTopic(model_name, model_type,num_labels=num_labels).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    for epoch in range(EPOCHS):
        print("EPOCH {} is training".format(epoch))
        train(model,training_loader,epoch,optimizer, num_labels=num_labels,one_hot=one_hot)
        torch.cuda.empty_cache()
    torch.save(model.state_dict(), model_path)
    del model
    return model_path




def train(model,training_loader,epoch,optimizer,num_labels, one_hot=False):

    scaler = torch.cuda.amp.GradScaler()
    model.train()
    data_size=len(training_loader)
    with tqdm(total=data_size) as pbar:
        for _, (data,label) in enumerate(training_loader, 0):
            
            input_ids = data['input_ids'].to(device).squeeze(1)
            attention_mask = data['attention_mask'].to(device).squeeze(1)
            if(one_hot):
                labels=torch.nn.functional.one_hot(torch.tensor(label),num_classes=num_labels).to(torch.float).to(device)
            else:
                labels = label.to(device)

            with torch.cuda.amp.autocast():
                output = model(input_ids = input_ids,attention_mask = attention_mask,labels=labels )
            
            loss=output.loss
            logits = output.logits

            pbar.update(1)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(loss)

def write_result(evaluation_hf,task,inference_time,model_name,model_type):
    path = '../topic_evaluation/perf_metrics.csv'.format(model_name)
    metadata_cols=['model_name','model_type','dataset','cost']
    if not os.path.exists(path):
        df = pd.DataFrame(columns=metadata_cols)
    else:
        df = pd.read_csv(path,index_col=0)
    result = {'model_name':model_name,'model_type':model_type,'dataset':'tweet_eval','cost':inference_time}
    
    for key,value in evaluation_hf.items():
        if isinstance(value,dict):
            for k,v in value.items():
                col_name = key+'_'+k
                result[col_name] = round(v,3)
        else:
            result[key] = round(value,3)
    df = df.append(result,ignore_index=True)
    f1_cols = [col for col in df.columns if 'f1' in col]
    metadata_cols.extend(f1_cols)
    df[metadata_cols].to_csv(path)

def foward_model(model,data,labels,num_labels,one_hot=False):
    input_ids = data['input_ids'].to(device).squeeze(1)
    attention_mask = data['attention_mask'].to(device).squeeze(1)
    if(one_hot):
        labels=torch.nn.functional.one_hot(torch.tensor(labels),num_classes=num_labels).to(torch.float).to(device)
    else:
        labels = labels.to(device)

    output = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
    loss=output.loss
    logits=output.logits
    logits = logits.detach().cpu().numpy()
    return logits,loss

def load_model(model_name,model_type,num_labels):
    model_path = "../topic_evaluation/{}-finetuned.pt".format(model_name.replace('/','-'))
    model = FinetuneClassTopic(model_name, model_type,num_labels=num_labels)
    model.load_state_dict(torch.load(model_path),strict=False)
    model.to(device)
    model.eval()
    return model


def validation_with_finetuned(model_name, model_type, dataset, num_labels,one_hot=False):
    model=load_model(model_name,model_type,num_labels)
    eval_loss = 0; eval_accuracy = 0; inference_time = 0
    n_correct = 0; n_wrong = 0; total = 0
    predictions , true_labels = [],[]
    nb_eval_steps, nb_eval_examples = 0, 0
    results_dict = {}
    val_loader = DataLoader(dataset, **val_params)
    with torch.no_grad():
        with tqdm(total=len(val_loader)) as pbar:
            for _, (data,labels) in enumerate(val_loader, 0):
                start = time.time()
                logits,loss=foward_model(model,data,labels,num_labels,one_hot)
                label_ids = labels.to('cpu').int().numpy()
                predictions.extend(np.argmax(logits,axis=1))
                if(one_hot):
                    true_labels.extend(np.argmax(label_ids,axis=1))
                else:
                    true_labels.extend(label_ids)
                eval_loss += loss.mean().item()
                inference_time += time.time() - start
                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1
                pbar.update(1)

        eval_loss = eval_loss/nb_eval_steps
        eval_cost = inference_time/nb_eval_steps
        logger.info("Validation loss: {}".format(eval_loss))
        logger.info("Validation cost: {}".format(eval_cost))

        from sklearn.metrics import classification_report
        eval_metrics = classification_report(true_labels, predictions,output_dict=True)
        write_result(eval_metrics,'eval_finetuned',eval_cost,model_name,model_type)
        del model
        return predictions, true_labels, eval_cost


def finetune(model_name, model_type, training_set, val_set, label_num):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset, val_dataset = data.getNewDataset(tokenizer,training_set,val_set)
    if model_type=='TF':
        model_path=pytorch_trainer(train_dataset,model_name,label_num,model_type,True)
        validation_with_finetuned(model_name, model_type, val_dataset,label_num,True)
    else:
        model_path=pytorch_trainer(train_dataset,model_name,label_num,model_type,False)
        validation_with_finetuned(model_name, model_type, val_dataset,label_num,False)
