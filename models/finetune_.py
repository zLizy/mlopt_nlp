import torch
import numpy as np
from datasets import Dataset
from datasets import load_dataset, load_metric
from transformers import TrainingArguments, Trainer, logging
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModelForTokenClassification
from pynvml import *
import bitsandbytes as bnb
from torch import nn
from transformers.trainer_pt_utils import get_parameter_names

# model_checkpoint = "distilbert-base-uncased"
batch_size = 16

default_args = {
    "output_dir": "train",
    "evaluation_strategy": "steps",
    "num_train_epochs": 1,
    "log_level": "error",
    "report_to": "none",
}

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

if __name__ == '__main__':

    # load data
    dataset_name = 'tweet_eval'
    dataset = load_dataset(dataset_name,split='test')

    addition = '_val'
    if 'val' in addition:
        dataset = dataset[:size]
    else:
        dataset = dataset[size:len(dataset)-1]

    metric = load_metric('tweet_eval')

    # load model
    model_name = 'elastic/distilbert-base-cased-finetuned-conll03-english'
    model = AutoModelForTokenClassification.from_pretrained(model_name).to("cuda")
    print_gpu_utilization()
    logging.set_verbosity_error()

    
    # ds_new = {}
    # ds_new['input_ids'] = ds['id']
    # ds_new['labels'] = ds['ner_tags']

    tokenizer = AutoTokenizer.from_pretrained('pretrained/{}'.format(model_name.replace('/','_')))
    def tokenize_function(examples):
        return tokenizer(examples["tokens"], padding="max_length", truncation=True)

    ds = dataset.map(tokenize_function, batched=True)
    # ds = Dataset.from_dict(ds_new)

    # Save memory
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,
        fp16=True,
        optim="adafactor",
        **default_args,
    )

    # trainer = Trainer(model=model, args=training_args, train_dataset=ds)
    # result = trainer.train()
    # print_summary(result)

    # Accelerate
    from accelerate import Accelerator
    from torch.utils.data.dataloader import DataLoader

    dataloader = DataLoader(ds, batch_size=training_args.per_device_train_batch_size)

    # if training_args.gradient_checkpointing:
    #     model.gradient_checkpointing_enable()

    # decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    # decay_parameters = [name for name in decay_parameters if "bias" not in name]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if n in decay_parameters],
    #         "weight_decay": training_args.weight_decay,
    #     },
    #     {
    #         "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
    #         "weight_decay": 0.0,
    #     },
    # ]

    # optimizer_kwargs = {
    #     "betas": (training_args.adam_beta1, training_args.adam_beta2),
    #     "eps": training_args.adam_epsilon,
    # }
    # optimizer_kwargs["lr"] = training_args.learning_rate
    # adam_bnb_optim = bnb.optim.Adam8bit(
    #     optimizer_grouped_parameters,
    #     betas=(training_args.adam_beta1, training_args.adam_beta2),
    #     eps=training_args.adam_epsilon,
    #     lr=training_args.learning_rate,
    # )

    # accelerator = Accelerator(fp16=training_args.fp16)
    # model, optimizer, dataloader = accelerator.prepare(model, adam_bnb_optim, dataloader)

    model.train()
    for step, batch in enumerate(dataloader, start=1):
        loss = model(**batch).loss
        loss = loss / training_args.gradient_accumulation_steps
        accelerator.backward(loss)
        if step % training_args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    # save
    path = 'finetuned/{}'.format(model_name.replace('/','_'))
    if not os.path.isdir(path):
        os.mkdir(path)

    tokenizer.save_pretrained(path)
    model.save_pretrained(path)


# # Vanilla training
# training_args = TrainingArguments(per_device_train_batch_size=2, **default_args)
# trainer = Trainer(model=model, args=training_args, train_dataset=ds)
# result = trainer.train()
# print_summary(result)

# # Gradient Accumulation
# training_args = TrainingArguments(per_device_train_batch_size=1, gradient_accumulation_steps=4, **default_args)
# trainer = Trainer(model=model, args=training_args, train_dataset=ds)
# result = trainer.train()
# print_summary(result)

# # Gradient Checkpointing
# training_args = TrainingArguments(
#     per_device_train_batch_size=1, gradient_accumulation_steps=4, optim="adafactor", gradient_checkpointing=True, **default_args
# )
# trainer = Trainer(model=model, args=training_args, train_dataset=ds)
# result = trainer.train()
# print_summary(result)

