from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/logger')
from logger.logger import get_logger
import torch
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

logger = get_logger(__name__, '../logs/Tokenizer.log', use_formatter=False)

 
# 'dslim/bert-base-NER','dslim/bert-base-NER-uncased',
# 'Jean-Baptiste/roberta-large-ner-english','Jean-Baptiste/camembert-ner-with-dates',
# 'Davlan/bert-base-multilingual-cased-ner-hrl','ml6team/bert-base-uncased-city-country-ner',
# 'Davlan/xlm-roberta-large-ner-hrl',
# 'jplu/tf-xlm-r-ner-40-lang','wolfrage89/company_segment_ner',
# 'Babelscape/wikineural-multilingual-ner','Davlan/xlm-roberta-base-ner-hrl',
# 'Davlan/distilbert-base-multilingual-cased-ner-hrl','ydshieh/roberta-large-ner-english',
# 'huggingface-course/bert-finetuned-ner','tner/tner-xlm-roberta-base-ontonotes5',
# 'chanifrusydi/bert-finetuned-ner','philschmid/distilroberta-base-ner-conll2003'
# 'NbAiLab/nb-bert-base-ner','kamalkraj/bert-base-cased-ner-conll2003'

model_name_dict = {'token':[
                            'tner/tner-xlm-roberta-base-ontonotes5',
                            
                            ],
                   'classifier':['flair/ner-english-large','flair/ner-english-ontonotes-large']}

class Tokenizer:
    def __init__(self, name, task='ner',framework='Pytorch',sentence='',):
        self.model_name = name
        self.sentence = sentence
        self.task = task
        self.framework = framework
        if self.task == 'pos':
            self.task = 'token-classification'

    def run():
        logger.info('======== {}: {} ========'.format(self.model_name, self.task))
        
        if self.task.lower() == 'ner':
            self.NERrun()

    def load_model(self):
        # device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained('pretrained/{}'.format(self.model_name.replace('/','_')))
        if self.framework.lower() == 'pytorch':
            model = AutoModelForTokenClassification.from_pretrained('pretrained/{}'.format(self.model_name.replace('/','_')))
        else:
            model = AutoModelForTokenClassification.from_pretrained('pretrained/{}'.format(self.model_name.replace('/','_')),from_tf=True)
        nlp = pipeline(self.task.lower(), model=model, tokenizer=tokenizer,aggregation_strategy="simple",device=0)
        return nlp

    def NERrun():
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if framework.lower() == 'pytorch':
            model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        else:
            model = AutoModelForTokenClassification.from_pretrained(self.model_name,from_tf=True)
        
        ner = pipeline(self.task.lower(), model=model, tokenizer=tokenizer,aggregation_strategy="simple")

        # Print results
        ner_results = ner(self.sentence)
        logger.into(ner_results)
  
  
if __name__ == '__main__':

    example = "My name is Wolfgang and I live in Berlin"
    model = Tokenizer(name='dslim/bert-base-NER',
                      sentence=example,
                      task='NER'
                    )
    model.run()
  