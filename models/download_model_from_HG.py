from huggingface_hub import hf_hub_download
from flair.models import SequenceTagger
from transformers import pipeline
# from neural_compressor.utils.load_huggingface import OptimizedModel
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
import pandas as pd
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/logger')
from logger.logger import get_logger

logger = get_logger(__name__, '../logs/HF_model_ner_new.log', use_formatter=False)

class HFModel():
    def __init__(self, task, config_file):
        self.task = task
        self.config = config_file

    def inference(self):
        for i, row in df.iloc[0:].iterrows():
            model_name = row['model_name']

    def download_model(self):
        logger.info('======= {} ======'.format(self.config))
        # df = pd.read_csv(self.config)
        df = pd.read_csv(self.config,index_col=0)
        logger.info(df.head())
        framework = []
        for i, row in df.loc[34:].iterrows():
            model_name = row['model_name']
            token = row['type']
            logger.info('model_name: {}, type: {}'.format(model_name,token))
            path = './pretrained/{}'.format(model_name.replace('/','_'))
            if token == 'token':
                framework.append('pytorch')
                # continue
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                try:
                    model = AutoModelForTokenClassification.from_pretrained(model_name)
                    framework.append('pytorch')
                except:
                    model = AutoModelForTokenClassification.from_pretrained(model_name,from_tf=True)
                    logger.info('-- a tensorflow model')
                    framework.append('tf')
                tokenizer.save_pretrained(path)
                model.save_pretrained(path)
            elif token == 'flair':
                # continue
                path = './pretrained/flair/{}'.format(model_name.replace('/','_'))
                framework.append('flair')
                tagger = SequenceTagger.load(model_name)
                tagger.save(model_file="{}.pt".format(path))
            elif token == 'pipeline':
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokenizer.save_pretrained(path)
                try:
                    model = AutoModelForSequenceClassification.from_pretrained(model_name)
                    framework.append('pytorch')
                except:
                    model = AutoModelForSequenceClassification.from_pretrained(model_name,from_tf=True)
                    logger.info('-- a tensorflow model')
                    framework.append('tf')
                model.save_pretrained(path)

            # hf_hub_download(repo_id=modelname, cache_dir="./pretrained/{}".format(modelname.replace('/','_')))
        if 'framework' not in df.columns:
            df['framework'] = framework
            df.to_csv(self.config)

if __name__ == '__main__':
    task = 'ner'
    # modify the name of the task and the weights will be saved under the ./pretrained folder
    
    CONFIG_FILE = 'model_{}.csv'.format(task)
    model = HFModel(task, CONFIG_FILE)
    
    # # download model files
    model.download_model()

