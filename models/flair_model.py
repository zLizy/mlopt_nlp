from flair.data import Sentence
from flair.models import SequenceTagger
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/logger')
from logger.logger import get_logger


# logger = get_logger(__name__, '../logs/Flair.log', use_formatter=False)

flair_model_list = ['flair/ner-english-large',
                    'flair/ner-english-fast','flair/ner-english',
                    'flair/ner-multi','flair/ner-multi-fast',
                    'flair/ner-english-ontonotes-large',
                    'flair/ner-english-ontonotes-fast','flair/ner-english-ontonotes']

class Flair:
    def __init__(self, name, sentence,task):
        self.model_name = name
        self.sentence = sentence
        self.task = task

    def run(self):
        # logger.info('======== {}: {} ========'.format(self.modelname, self.task))
        
        if self.task.lower() == 'ner':
            self.NERrun()

    def load_model(self):
        tagger = SequenceTagger.load('pretrained/flair/{}.pt'.format(self.model_name.replace('/','_')))
        return tagger
    
    def NERrun(self):
        # load the NER tagger
        tagger = SequenceTagger.load(self.modelname)
        # run NER over sentence
        sentence = Sentence(self.sentence)
        tagger.predict(sentence)
        # iterate over entities and print each
        # for entity in sentence.get_spans(self.task.lower()):
        #     logger.info('- {}'.format(entity))



if __name__ == '__main__':

    example = "My name is Wolfgang and I live in Berlin"
    model = Flair(name='flair/ner-english-fast',
                      sentence=example,
                      task='NER'
                 )
    model.run()