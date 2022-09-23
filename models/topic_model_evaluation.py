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

from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'

def load_tweet():
    corpus=load_dataset('tweet_eval','sentiment',split='train')['text']
    # with open(data_path) as json_file:
    #     for j_str in json_file:
    #         text=json.loads(j_str)['text']
    #         corpus.append(text)
    return corpus

def load_yahoo():
    dataset = load_dataset('yahoo_answers_topics', split='test')
    size=len(dataset)//6
    return dataset['best_answer'][:size], dataset['topic'][:size]

def load_truth(train=False):
    corpus=load_tweet()
    if(train):
        labels=np.load('../topic_evaluation/ground_truth.npy')
    else:
        labels=np.zeros(len(corpus))
    return corpus,labels

def record_prediction(model_name, prediction_nparray):
    evaluation_result = '../topic_evaluation'
    if not os.path.isdir(evaluation_result):
        os.mkdir(evaluation_result)
    np.save('../topic_evaluation/ground_truth.npy'.format(model_name),prediction_nparray)
    
def write_result(evaluation_hf,task,inference_time,model_name,model_type):
    path = '../topic_evaluation/evaluation.csv'
    if not os.path.exists(path):
        df = pd.DataFrame(columns=['model_name','model_type','task','dataset','cost'])
    else:
        df = pd.read_csv(path,index_col=0)
    result = {'model_name':model_name,'model_type':model_type,'task':task,'dataset':'yahoo_answers_topics','cost':inference_time}
    
    for key,value in evaluation_hf.items():
        if isinstance(value,dict):
            for k,v in value.items():
                col_name = key+'_'+k
                result[col_name] = round(v,3)
        else:
            result[key] = round(value,3)
    df = df.append(result,ignore_index=True)
    df.to_csv(path)

def evaluate(model_name,model_type, procedure='inference', record_prediction=False):
    if(procedure=='inference'):
        corpus,label=load_yahoo()
    elif(procedure=='ground_truth'):
        corpus,label=load_truth()
    else:
        corpus,label=load_truth(True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_type=='TF':
        model = AutoModelForSequenceClassification.from_pretrained(model_name,from_tf=True).to(device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    y_pred=[]
    y_true=[]
    with tqdm(total=len(corpus)) as pbar:
        for i,text in enumerate(corpus):
            tokens = tokenizer(text, return_tensors='pt',truncation=True).to(device)
            if(tokens['input_ids'].size(dim=1)>512):
                y_pred.append(-100)
                y_true.append(label[i])
                continue
            output=model(**tokens)
            scores=output[0][0].cpu().detach().numpy()
            scores=expit(scores)
            prediction=np.argmax(scores)
            y_pred.append(prediction)
            y_true.append(label[i])
            pbar.update(1)
    if(procedure=='ground_truth'):
        record_prediction(model_name,np.array(y_pred,dtype='int'))
    from sklearn.metrics import classification_report
    eval_metrics = classification_report(y_true, y_pred,output_dict=True)
    write_result(eval_metrics,procedure,0.01,model_name,model_type)


def select_best_model():
    df=pd.read_csv('../topic_evaluation/evaluation.csv')
    best_item=df.sort_values('macro avg_f1-score',ascending=False).iloc[0]
    return best_item['model_name'],best_item['model_type']
    



