ner_labels = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
ner_map = {'LABEL_0':'O','LABEL_1':'B-PER','LABEL_2':'I-PER','LABEL_3':'B-ORG','LABEL_4':'I-ORG','LABEL_5':'B-LOC','LABEL_6':'I-LOC','LABEL_7':'B-MISC','LABEL_8':'I-MISC'}
num2ner = {v:k for k, v in ner_labels.items()}
for k,v in num2ner.items():
    if v != 'O':
        if 'I-' in v:
            num2ner[k] = v.replace('I-','B-')

ner_labels = {'O': 0, 'B-PER': 1, 'B-ORG': 2, 'B-LOC': 3, 'B-MISC': 4}

pos_tag = {'"': 0, "''": 1, '#': 2, '$': 3, '(': 4, ')': 5, ',': 6, '.': 7, ':': 8, '``': 9, 'CC': 10, 'CD': 11, 'DT': 12,
            'EX': 13, 'FW': 14, 'IN': 15, 'JJ': 16, 'JJR': 17, 'JJS': 18, 'LS': 19, 'MD': 20, 'NN': 21, 'NNP': 22, 'NNPS': 23,
            'NNS': 24, 'NN|SYM': 25, 'PDT': 26, 'POS': 27, 'PRP': 28, 'PRP$': 29, 'RB': 30, 'RBR': 31, 'RBS': 32, 'RP': 33,
            'SYM': 34, 'TO': 35, 'UH': 36, 'VB': 37, 'VBD': 38, 'VBG': 39, 'VBN': 40, 'VBP': 41, 'VBZ': 42, 'WDT': 43,
            'WP': 44, 'WP$': 45, 'WRB': 46}
num2pos = {v:k for k, v in pos_tag.items()}
# https://universaldependencies.org/tagset-conversion/en-penn-uposf.html
pos_map = {'"':'PUNCT', "''":'PUNCT', '#': 'SYM',  '$': 'SYM', '(':'PUNCT', ')':'PUNCT', ',': 'PUNCT', '.':'PUNCT', ':':'PUNCT', '``':'PUNCT',
            'CC':'CCONJ', 'CD': 'NUM','DT':'DET', 'EX': 'PRON', 'FW':'X', 'IN':'ADP', 'JJ':'ADJ', 'JJR':'ADJ','JJS':'ADJ', 'LS': 'X',
            'MD': 'VERB', 'NN': 'NOUN', 'NNP': 'PROPN', 'NNPS': 'PROPN', 'NNS': 'NOUN', 'NN|SYM':'SYM', 'PDT':'DET', 'POS':'PART',
            'PRP':'PRON', 'PRP$':'DET','RB':'ADV','RBR':'ADV', 'RBS':'ADV', 'RP':'ADP', 'SYM':'SYM', 'TO':'PART', 'UH':'INTJ','AUX':'AUX','SCONJ':'SCONJ',
            'VB':'VERB', 'VBD':'VERB', 'VBG':'VERB', 'VBN':'VERB', 'VBP':'VERB', 'VBZ':'VERB', 'WDT':'DET', 'WP':'PRON', 'WP$':'DET', 'WRB':'ADV',
            '-LRB-':'PUNCT','-RRB-':'PUNCT'}
num2pos = {k:pos_map[v] for k,v in num2pos.items()}
# pos2num = 

# sentiment analysis
sentiment_tag = {'label_0':0,'label_1':1,'label_2':2,'label_3':2,'negative':0,'positive':2,'neutral':1,'pos':2,'neg':0,'neu':1,
                 '1 star':0,'2 stars':0,'3 stars':1,'4 stars':2,'5 stars':2}

num2sentiment = {0: 'negative', 1: 'neutral', 2: 'positive'}
# num2pos = {v:k for k,v in sentiment_tag.items()}

num2topic = {0: 'arts_&_culture',	5: 'fashion_&_style',	10: 'learning_&_educational',	15: 'science_&_technology',
             1: 'business_&_entrepreneurs',	6: 'film_tv_&_video',	11: 'music',	16: 'sports',
             2: 'celebrity_&_pop_culture',	7: 'fitness_&_health',	12: 'news_&_social_concern',	17: 'travel_&_adventure',
             3: 'diaries_&_daily_life',	8: 'food_&_dining',	13: 'other_hobbies',	18: 'youth_&_student_life',
             4: 'family',	9: 'gaming',	14: 'relationships'}
topic_tag = {v:k for k,v in num2topic.items()}

# label_list = ['B-O', 'B-PER', 'B-ORG', 'B-LOC', 'B-MISC']
label_list = {'pos':[ 'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X'],
              'ner': list(ner_labels.keys()),
              'ner_ori':['O', 'B-PER', 'B-PER', 'B-ORG', 'B-ORG', 'B-LOC', 'B-LOC', 'B-MISC','B-MISC']
              }

label2num = {'ner':{'O': 0, 'B-PER': 1, 'B-ORG': 2, 'B-LOC': 3, 'B-MISC': 4},
             'pos': {k:i for i,k in enumerate(label_list['pos'])}}