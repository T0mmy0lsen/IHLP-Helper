from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TFAutoModelForSequenceClassification, AutoConfig, \
    BertModel, RobertaForSequenceClassification, TFRobertaForSequenceClassification
from transformers import create_optimizer
from pipeline.models.loader import Loader

import pandas as pd
import numpy as np
import tensorflow as tf

import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# num_labels = 1      # Regression
# num_labels = 2      # For keep_else
num_labels = 100    # For top_100

# model = TFAutoModelForSequenceClassification.from_pretrained("saved/model_bert_by_example/", num_labels=num_labels)

_, _, x_train, y_train, x_test, y_test, id2label, label2id = Loader.get_word_embedding_and_labels(labels='top_100')

config = AutoConfig.from_pretrained("xlm-roberta-base")
config.hidden_dropout_prob = 0.15
# config.id2label = id2label
# config.label2id = label2id

model = TFRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=num_labels)
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

x_train = [e[:512] if isinstance(e, str) else '' for e in x_train]
x_test = [e[:512] if isinstance(e, str) else '' for e in x_test]

df = pd.DataFrame(np.stack((x_train, y_train), axis=1), columns=['text', 'label'])
df.to_csv('cache/train.csv', index=False)

df = pd.DataFrame(np.stack((x_test, y_test), axis=1), columns=['text', 'label'])
df.to_csv('cache/test.csv', index=False)

data_files = {"train": "cache/train.csv", "test": "cache/test.csv"}
dataset = load_dataset("csv", data_files=data_files)

def preprocess_function(data):
    return tokenizer(data['text'], truncation=False)

tokenized_imdb_train = dataset['train'].map(preprocess_function, batched=True)
tokenized_imdb_test = dataset['test'].map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf", max_length=512)

tf_train_set = model.prepare_tf_dataset(
    tokenized_imdb_train,
    shuffle=True,
    batch_size=2,
    collate_fn=data_collator,
)

tf_validation_set = model.prepare_tf_dataset(
    tokenized_imdb_test,
    shuffle=False,
    batch_size=1,
    collate_fn=data_collator,
)

batch_size = 2
num_epochs = 1
batches_per_epoch = len(tokenized_imdb_train) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=10000, num_train_steps=total_train_steps)

model.compile(optimizer=optimizer, metrics=['accuracy'])
model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=num_epochs)

model.save_pretrained('saved/model_bert_by_example/', overwrite=True)