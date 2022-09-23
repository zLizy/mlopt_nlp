
import os
import torch
import pandas as pd
import numpy as np
from config import *
from util import *
from inference import *
from datasets import load_dataset, load_metric
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler


def remove_nan(list_of_lists):
    return [[l for l in lists if pd.isnull(l)==False] for lists in list_of_lists]

def load_data(task,dataset_name,addition):
    if dataset_name == 'conll2003':
        dataset = load_dataset(dataset_name, split='train')
        dataset_path = '../train/{}_{}'.format(dataset_name,task)
    elif dataset_name == 'tweet_eval':
        dataset = load_dataset(dataset_name, 'sentiment', split='train')
        dataset_path = '../train/{}_{}'.format(dataset_name,task)
    elif dataset_name == 'glue':
        dataset = load_dataset(dataset_name, task, split='validation')
    
    if not os.path.isdir(dataset_path):
        os.makedirs(dataset_path)
    
    return dataset

def get_ground_truth(dataset_name,task):
    gt_save_path = '../train/{}_{}/gt_.csv'.format(dataset_name,task)
    if not os.path.exists(gt_save_path):
        data = load_data(task,dataset_name,'')
        # write_gt(dataset,path,task,purpose,dataset_name)
        write_gt(data,gt_save_path,task,'experiment',dataset_name)
        # write_gt(dataset,path,task,purpose,dataset_name)
        # df.to_csv(gt_save_path)
        # _df = pd.read_csv(gt_save_path,index_col=0)
        # df = pd.DataFrame()
        # df['labels'] = remove_nan(_df[_df.columns[:]].values.tolist())
        # df['text'] = data['text']
        # df.to_csv(gt_save_path)
    
    df = pd.read_csv(gt_save_path,index_col=0)
    # if 'label' in df.columns:
    
    data = load_data(task,dataset_name,'')
    df_ = pd.DataFrame()
    df_['labels'] = remove_nan(df[df.columns[:]].values.tolist())
    df_['labels'] = [[label2num[task][l] for l in label] for label in df_['labels']]# df_.replace({"labels": ner_labels})
    df_['text'] = data['text']
    df_.to_csv('../train/{}_{}/gt.csv'.format(dataset_name,task))
    df = df_
            
    return df

def createDataset(df,train_size=0.7):
    # Creating the dataset and dataloader for the neural network
    train_dataset=df.sample(frac=train_size,random_state=200)
    if(train_size>0.9999):
        val_dataset=None
    else:
        val_dataset=df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    # train_encodings = tokenizer(list(train_dataset['text']), truncation=True,padding=True, is_split_into_words=True)
    # logger.info('tokenized encodings keys: {}'.format(train_encodings.keys()))
    # logger.info('tokenized encodings input_ids len: {}'.format(len(train_encodings['input_ids'])))
    # logger.info('tokenized encodings input_ids: {}'.format(train_encodings['input_ids'][:2]))
    # val_encodings = tokenizer(list(val_dataset['text']), truncation=True, padding=True, is_split_into_words=True)

    # train_labels = tokenize_and_align_labels(train_encodings,list(train_dataset['labels']))
    # logger.info('tokenized train_dataset[labels]: {}'.format(train_labels[:5]))
    # val_labels = tokenize_and_align_labels(val_encodings,list(val_dataset['labels']))

    # training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
    # testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)
    
    return train_dataset, val_dataset

def getCustomDataset(tokenizer,training_set,val_set):

    train_dataset = IMDbDataset(tokenizer, list(training_set['text']), list(training_set['labels']), MAX_LEN) #(train_encodings, train_labels)
    val_dataset = IMDbDataset(tokenizer, list(val_set['text']), list(val_set['labels']), MAX_LEN) #(val_encodings, val_labels)

    return train_dataset, val_dataset

def getNewDataset(tokenizer,training_set,val_set):

    train_dataset = newDataset(tokenizer, list(training_set['text']), list(training_set['labels'])) #(train_encodings, train_labels)
    val_dataset = newDataset(tokenizer, list(val_set['text']), list(val_set['labels'])) #(val_encodings, val_labels)

    return train_dataset, val_dataset


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


def tokenize_and_align_labels(tokenizer,sentences,labels):
    tokenized_inputs = tokenizer(
        [sentence.split() for sentence in sentences], truncation=True, is_split_into_words=True, padding=True
    )
    all_labels = labels
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs



def getTokenizedDataset(tokenizer,training_set,val_set):
    # train_encodings = tokenizer([s.split() for s in list(training_set['text'])], is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)
    # val_encodings = tokenizer([s.split() for s in list(val_set['text'])], is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)
    # logger.info('train_encodings.keys: {}'.format(train_encodings.keys()))
    # logger.info('train_encodings.word_ids: {}'.format(train_encodings.word_ids()[0]))

    # logger.info('labels: {}'.format(list(training_set['labels'])[0]))
    # train_labels = encode_tags(list(training_set['labels']), train_encodings)
    # val_labels = encode_tags(list(val_set['labels']), val_encodings)

    train_encodings = tokenize_and_align_labels(tokenizer,list(training_set['text']),list(training_set['labels']))
    val_encodings = tokenize_and_align_labels(tokenizer,list(val_set['text']),list(val_set['labels']))

    # train_encodings.pop("offset_mapping") # we don't want to pass this to the model
    # val_encodings.pop("offset_mapping") 

    train_dataset = XDataset(train_encodings, train_encodings['labels'])
    val_dataset = XDataset(val_encodings, val_encodings['labels'])

    return train_dataset, val_dataset


######### Dataset Class

class newDataset(torch.utils.data.Dataset):
    def __init__(self,tokenizer,data,label):
        self.tokenizer=tokenizer
        self.data=data
        self.label=label
    def __getitem__(self,idx):
        token= self.tokenizer(self.data[idx], return_tensors='pt',padding='max_length',max_length=512,truncation=True)
        l=self.label[idx]
        return token,l
    
    def __len__(self):
        return len(self.label)

### create pytorch dataset object
class XDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.as_tensor(val[idx]) for key, val in self.encodings.items()}
        #item = {key: torch.as_tensor(val[idx]).to(device) for key, val in self.encodings.items()}
        # item['labels'] = torch.as_tensor(self.labels[idx])
        #item['labels'] = self.labels[idx]
        return item
    def __len__(self):
        return len(self.labels)

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.comment_text
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.int),
            'mask': torch.tensor(mask, dtype=torch.int),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.int),
            'targets': torch.tensor(self.targets[index], dtype=torch.int)
        }

import numpy as np

def encode_tags(labels, encodings):
    # labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        # logger.info('doc_offset: {}'.format(len(doc_offset)))
        # logger.info('label: {}'.format(doc_labels))
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)
        # logger.info('arr_offset: {}'.format((arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)))

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels



class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, sentences, labels, max_len):
        self.len = len(sentences)
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

        # self.encodings = encodings
        self.labels = labels

    # def __getitem__(self, idx):
    #     item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    #     item['labels'] = torch.tensor(self.labels[idx])
    #     return item

    def __getitem__(self, idx):
        # print(type(idx),idx)
        sentence = self.sentences[idx]
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )

        # input_ids = torch.tensor(self.encodings['input_ids'],dtype=torch.long)
        input_ids = torch.tensor(inputs['input_ids'],dtype=torch.int)
       
        mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)

        label = self.labels[idx]
        label.extend([0]*self.max_len)
        label=label[:self.max_len]
        target_ids = torch.tensor(label,dtype=torch.int)

        return {"input_ids": input_ids, "mask": mask, "labels": target_ids}

    def __len__(self):
        return len(self.labels)

