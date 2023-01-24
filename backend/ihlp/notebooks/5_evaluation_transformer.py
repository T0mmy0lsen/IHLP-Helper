import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin")

import pandas as pd
import tensorflow as tf

from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

LABEL = 'label_responsible'
LABEL_MODEL = 'Responsible'

model = TFAutoModelForSequenceClassification.from_pretrained(f'data/models/IHLP-XLM-RoBERTa-{LABEL_MODEL}')
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

df = pd.read_csv(f'data/cached_test_{LABEL}.csv')
df = df[-1000:]

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
results = model.evaluate(tokenized_text, df.label.values, batch_size=16, verbose=False)

print(results)