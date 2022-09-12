# Optimization of Machine Learning Inferece Query on NLP Tasks

## Dataset

[Tweet_eval](https://huggingface.co/datasets/tweet_eval)
- originated for sentiment analysis
- also served for other tasks

## Tasks
- Name Entity Recognition; 
- Sentiment Analysis; 
- Topic Classification (Multi-label/Multi-class)

## Steps

### 1. Model basic info collection (HuggingFace)
- Determine the task
- Identify the models that solve the task and verify the labels
- Download the weight files for efficiency
	```
	python3 download_model_from_HG.py
	```
	```
	if __name__ == '__main__':
	    task = 'ner' 
	    # modify the name of the task and the weights will be saved under the ./pretrained folder
	    
	    CONFIG_FILE = 'model_{}.csv'.format(task)
	    model = HFModel(task, CONFIG_FILE)
	    
	    # # download model files
	    model.download_model()
	```

### 2. Evaluate the collected models on its training dataset (validation/test set) 
- Evaluate the performance in terms of accuracy, f1 score, precision, recall;
- Evaluate the performance in terms of efficiency, i.e., execution speed per instance
	```
	python3 inference.py
	```

	```
	if __name__ == '__main__':
	    task =  'ner' #'sentiment' #'ner' #'pos'
	    purpose = 'experiment' # 'experiment' 'evaluate'
	    # evaluate: evaluate all the models on its origin dataset
	    # experiment: evaluate the models on dedicated dataset with groundtruth being retrieved

	    dataset_name = 'tweet_eval' #'tweet_eval'#'conll2003'
	    addition = '_val' #'_val' # ''
	    # val: validation set
	    # test: test set

	    ## results will be saved to '../evaluation/{}_{}_{}_results.csv'.format(dataset_name,addition,task)
	```

### 3. Retrieve ground truth on the dedicated dataset (tweet_eval) for different tasks
- Identify the best-performaing model as the reference model
- Execute the best model on **tweet_eval** and record its predictions as ground truth
	```
	# Modify this function in inference.py by adding topic classification task (topic)
	def write_gt(dataset,path,task,purpose,dataset_name):
	    if not os.path.exists(path):
	        # evaluate: model inference
	        if purpose == 'evaluate':
	            if task == 'sentiment':
	                tag_list = dataset['label']
	            else:
	                tag_list = dataset['{}_tags'.format(task)]
	            convert_label(path,tag_list,task)
	        # experiment: gather gt and evaluate on SOA models
	        if purpose == 'experiment':
	            if task == 'ner':
	                model_name = 'flair/ner-multi'
	                m_type = 'flair'
	            elif task == 'pos':
	                model_name = 'flair/upos-english'
	                m_type = 'flair'
	            elif task == 'sentiment':
	                model_name = 'cardiffnlp/twitter-roberta-base-sentiment'
	                m_type = 'pipeline'
	            model = load_model(model_name,m_type,task,'pytorch')
	            inference_time = inference(model,dataset,task,path,m_type,dataset_name)
	```
	```
	python3 new_finetune.py
	# new_finetune.py will retrieve the above function before finetuning it
	# new_finetune include step 3 & 4
	```

### 4. Finetune the rest of the models for a certain task on the new ground truth
- Finetune the models
- Evaluate the performance on **tweet_eval**
	```
	python3 new_finetune.py
	```
	```
	if __name__ == '__main__':

	    task =  'ner' #'sentiment' #'ner' #'pos'
	    purpose = 'evaluate' # 'experiment' 'evaluate'
	    finetune = False

	    dataset_name = 'tweet_eval' #'tweet_eval'#'conll2003'
	    addition = 'train' #'_val' # ''

	    data = get_ground_truth(dataset_name,task)
	    training_set, val_set = createDataset(data)
	```


