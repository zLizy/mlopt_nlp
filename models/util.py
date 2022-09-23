import torch
from GPUtil import showUtilization as gpu_usage
import pandas as pd
from datasets import load_metric
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModelForTokenClassification, PreTrainedModel, AutoConfig,AutoTokenizer,AutoModelForSequenceClassification
from torch import cuda
import evaluate
import torch
device = 'cuda:0' if cuda.is_available() else 'cpu'
import accelerate
import time
from pynvml import *
from config import *

# def print_summary(result):
#     logger.log(f"Time: {result.metrics['train_runtime']:.2f}")
#     logger.log(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
#     print_gpu_utilization()

# import bitsandbytes

def print_gpu_utilization(logger):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    logger.info(f"GPU memory occupied: {info.used//1024**2} MB.")

def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()                             
    # cuda.close()
    # cuda.select_device(0)
    torch.cuda.empty_cache()
    # cuda.select_device(0)
    print("GPU Usage after emptying the cache")
    gpu_usage()

def convert_to_b(y):
    """Append "B-" to start of all tags"""
    return [
        [f'B-{tag}' for tag in tags]
        for tags in y
    ]


# load tokenizer
def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained('pretrained/{}'.format(model_name.replace('/','_')),add_prefix_space=True)
    # model = AutoModelForTokenClassification.from_pretrained('pretrained/{}'.format(model_name.replace('/','_'))).to(device)
    # logger.info(torch.cuda.memory_summary(device=None, abbreviated=False))
    return tokenizer


import numpy as np

# Data Processing for Token Classification
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

def eval(predictions,labels,task):
    # metric
    if 'ner' in task:
        metric = evaluate.load('seqeval')
    elif 'pos' in task:
        metric = evaluate.load('poseval')

    results = metric.compute(predictions=predictions, references=labels)
    return results

def compute_metrics(p,task):
    # label_list = ['B-O', 'B-PER', 'B-ORG', 'B-LOC', 'B-MISC']

    predictions, labels = p
    predictions = np.argmax(np.array(predictions), axis=2)
    labels = np.array(labels,dtype=object)
    print(predictions.shape,labels.shape,type(labels),type(labels[0]))
    # labels = labels.reshape(predictions.shape[0],predictions.shape[1])
    
    # print(predictions.shape,labels.shape)
    # print([(prediction, label) for prediction, label in zip(predictions[0], labels[0])])

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[task][p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[task[:-4]][l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = eval(true_predictions,true_labels,task)
    return results
    # return {
    #     "precision": results["overall_precision"],
    #     "recall": results["overall_recall"],
    #     "f1": results["overall_f1"],
    #     "accuracy": results["overall_accuracy"],
    # }

 ####### Sections of config
# Defining some key variables that will be used later on in the training
MAX_LEN = 50
TRAIN_BATCH_SIZE = 50
VALID_BATCH_SIZE = 50
EPOCHS = 10
LEARNING_RATE = 1e-03
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0,
                }

val_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0,
              }

def generate_from_model(model,tokenizer):
    encoded_input = tokenizer(text,return_tensors='pt')

######### Custom Model
class FinetuneClass(PreTrainedModel):
    def __init__(self,model_name,task,num_labels,freeze_encoder=False):
        model_path = './pretrained/{}'.format(model_name.replace('/','_'))
        # super(FinetuneClass, self).__init__()
        config = AutoConfig.from_pretrained(model_name)
        super().__init__(config)
        torch.cuda.empty_cache()
        try: #TODO sequenceClassification
            self.l1 = AutoModelForTokenClassification.from_pretrained(model_name,device_map='auto', load_in_8bit=True).to(device)
        except:
            print('device_map not working')
            self.l1 = AutoModelForTokenClassification.from_pretrained(model_name).to("cpu")#.to(device)# torch_dtype=torch.float16
            torch.cuda.empty_cache()
        # AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
        in_features = self.l1.classifier.in_features
        print('in_features',in_features)
        self.l1.classifier = torch.nn.Linear(in_features,num_labels).to(device)
        self.l1.num_labels = num_labels
        # model.config.num_labels = 2
        # self.l2 = torch.nn.Dropout(0.3)
        # self.l3 = torch.nn.Linear(768, 5)
    
    def forward(self, input_ids = None, attention_mask = None, labels = None):
        output_1= self.l1(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
        # output_2 = self.l2(output_1[0])
        # output = self.l3(output_2)
        return output_1

class FinetuneClassTopic(PreTrainedModel):
    def __init__(self,model_name,model_type,num_labels,freeze_encoder=False):
        # super(FinetuneClass, self).__init__()
        config = AutoConfig.from_pretrained(model_name)
        super().__init__(config)
        torch.cuda.empty_cache()
        if model_type=='TF':
            self.l1 = AutoModelForSequenceClassification.from_pretrained(model_name,from_tf=True).to(device)
        else:
            self.l1 = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        if(model_name.startswith('cardiffnlp') 
            or model_name in ['boronbrown48/1_model_topic_classification_v2','dhtocks/Topic-Classification','boronbrown48/wangchanberta-topic-classification'] ):
            self.l1.classifier.out_proj=torch.nn.Linear(in_features=768,out_features=num_labels,bias=True).to(device)
        else:
            in_features = self.l1.classifier.in_features
            self.l1.classifier = torch.nn.Linear(in_features,num_labels).to(device)
        self.l1.num_labels = num_labels
        # model.config.num_labels = 2
        # self.l2 = torch.nn.Dropout(0.3)
        # self.l3 = torch.nn.Linear(768, 5)
    
    def forward(self, input_ids = None, attention_mask = None, labels = None):
        output_1= self.l1(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
        # output_2 = self.l2(output_1[0])
        # output = self.l3(output_2)
        return output_1
    

