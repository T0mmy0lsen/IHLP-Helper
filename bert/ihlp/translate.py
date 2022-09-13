from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import pandas as pd
from tqdm import tqdm


class Translate:

    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-da-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-da-en")

    def __init__(self, file=None, top=None):
        if file is not None:
            df = pd.read_csv(file)
            if top is not None:
                top_list = df['label'].value_counts().index.tolist()
                df = df[df['label'].isin(top_list[:100])]
            self.df = df
            self.run()

    def run(self):

        def make_file(folder, d, translate):
            path = f'{folder}/{d.label}/{d.id}.txt'
            if not os.path.exists(path):
                with open(f'{folder}/{d.label}/{d.id}.txt', 'w') as f:
                    text = d['text']
                    text = translate.translate(text)
                    text = text.lower()
                    f.write(text)

        def f(label):
            df_tmp = self.df[self.df['label'] == label]
            translate = Translate()
            for i, d in tqdm(df_tmp.iterrows()):
                make_file('translated', d, translate)

        labels = self.df['label'].unique()

        for label in labels:
            print("Creating for: {}".format(label))
            f(label)

    def translate(self, text):
        input_ids = self.tokenizer(text[:512], return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids=input_ids, num_beams=4, num_return_sequences=1, max_length=512)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

Translate(file='request_responsible_da.csv', top=100)