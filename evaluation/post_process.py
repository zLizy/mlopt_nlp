import pandas as pd

tasks = ['ner','pos','sentiment']
metrics = ['precision','recall','f1','accuracy']

# model_name,task,dataset,cost,ADD_f1,ADD_number,ADD_precision,
for metric in metrics:
    result_path = '{}.csv'.format(metric)
    df_list = []
    print('====== {} ======'.format(metric))
    for task in tasks:
        path = 'tweet_eval_{}_results.csv'.format(task)
        print(path)

        df = pd.read_csv(path,index_col=0)
        # print(df.head())
        columns = df.columns
        distinct_label = list(set([c for c in columns if '_'+metric in c and 'overall' not in c and 'avg' not in c]))
        
        new_columns = ['model_name','cost']+distinct_label
        print(new_columns)
        df = df[new_columns]
        df.columns = [l.replace('-score','').replace('1_','neutral_').replace('2','positive').replace('0','negative').replace('_'+metric,'') for l in new_columns]
        df_list.append(df)

    result = pd.concat(df_list,ignore_index=True)
    result.to_csv(result_path)