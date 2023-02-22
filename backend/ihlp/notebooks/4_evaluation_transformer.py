import os
import psutil as psutil
import time

# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin")

import pandas as pd
import tensorflow as tf

from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

LABEL = 'label_placement'
LABEL_MODEL = 'Placement-Subject'

model = TFAutoModelForSequenceClassification.from_pretrained(f'data/models/IHLP-XLM-RoBERTa-{LABEL_MODEL}')
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

df = pd.read_csv(f'data/cached_test_subject_{LABEL}.csv')
df = df[-5:]

print(df.head())


def tokenize_texts(sentences, max_length=512, padding='max_length'):
    return tokenizer(
        sentences,
        truncation=False,
        padding=padding,
        max_length=max_length,
        return_tensors="tf"
    )

tokenized_text = dict(tokenize_texts(list(df['text'].values)))

model.compile(metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3)])
print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

start_time = time.time()
results = model.evaluate(tokenized_text, df.label.values, batch_size=1, verbose=False)

print("--- %s seconds ---" % (time.time() - start_time))
print(results)