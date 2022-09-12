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

def load_token_model(model_name,task,framework):
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained('pretrained/{}'.format(model_name.replace('/','_')))
    if framework.lower() == 'pytorch':
        from_tf = False
    else:
        from_tf = True
    if task == 'pos':
        task = 'token-classification'
    model = AutoModelForTokenClassification.from_pretrained('pretrained/{}'.format(model_name.replace('/','_')),from_tf=from_tf)
    nlp = pipeline(task.lower(), model=model, tokenizer=tokenizer,aggregation_strategy="simple",device=0)
    # int8_model = OptimizedModel.from_pretrained('Intel/distilbert-base-uncased-finetuned-conll03-english-int8-static',)
    return nlp

def load_model(model_name,m_type,task,framework):
    if m_type == 'token':
        model = load_token_model(model_name,task,framework)
        # model = Tokenizer(model_name,task=task,framework=framework).load_model()
        return model
    elif m_type == 'flair':
        model = SequenceTagger.load('pretrained/flair/{}.pt'.format(model_name.replace('/','_')))#Flair(model_name,'',task=task).load_model()
        return model
    elif m_type == 'pipeline':
        model = load_pipeline_model(model_name,task,framework)
        return model

dataset_name = 'conll2013'
model_name = ''

dataset = load_dataset(dataset_name, split='test')
s1 = dataset['tokens'][0]
results = model(s1)