from itertools import islice

import pandas as pd
import torch
import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from ftlangdetect import detect
from tqdm import tqdm

tqdm.pandas()
pd.options.mode.chained_assignment = None

PATH_FROM_CLEANING = 'data/output/2_data_cleaning/output_heavy.csv'

PATH_OUTPUT_CHECKPOINT_TRANSLATE_DETECT = 'data/output/3_data_translate/output_0_detect.csv'
PATH_OUTPUT_CHECKPOINT_TRANSLATE_EN = 'output_en.csv'

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

# pip uninstall torch
# pip cache purge
# pip install torch -f https://download.pytorch.org/whl/torch_stable.html


class DataTranslate:

    df_detected = None

    def __init__(self, debug=True):

        self.df = pd.read_csv(PATH_FROM_CLEANING, sep=",", quotechar="\"", dtype=str)
        print("[Cleaning] Loading data completed.")

        if debug:
            self.df = self.df.head(100)

        self.run()


    def checkpoint_detect(self):

        print('Running checkpoint_detect()')

        tmp = self.df.fillna('').copy()

        tmp['subject_lang'] = tmp.progress_apply(lambda x: detect(x['subject'], low_memory=False)['lang'], axis=1)
        tmp['subject_translated'] = False

        tmp['description_lang'] = tmp.progress_apply(lambda x: detect(x['description'], low_memory=False)['lang'], axis=1)
        tmp['description_translated'] = False

        self.df_detected = tmp
        tmp.to_csv(PATH_OUTPUT_CHECKPOINT_TRANSLATE_DETECT, index=False)


    def checkpoint_translate(self, to_lang='en', from_lang='da', index_lang='da', path_output=PATH_OUTPUT_CHECKPOINT_TRANSLATE_EN):

        print(f'Running checkpoint_translate(to_lang={to_lang}, from_lang={from_lang}, index_lang={index_lang}, path_output={path_output})')
        self.df_detected = pd.read_csv(PATH_OUTPUT_CHECKPOINT_TRANSLATE_DETECT)

        tokenizer = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{from_lang}-{to_lang}")
        model = AutoModelForSeq2SeqLM.from_pretrained(f"Helsinki-NLP/opus-mt-{from_lang}-{to_lang}")
        model = model.to(device)

        tmp = self.df_detected.fillna('')

        def translate(text_index):

            def handle(texts, texts_index, texts_length):

                texts_tokenizer = tokenizer(texts, return_tensors="pt", padding=True).to(device).input_ids
                texts_model = model.generate(input_ids=texts_tokenizer, num_beams=4, max_length=512)
                texts_out = tokenizer.batch_decode(texts_model, skip_special_tokens=True)

                text_batch = []
                text_batch_index = 0

                for j in texts_length:
                    text_batch.append(texts_out[text_batch_index:text_batch_index + j])
                    text_batch_index += j

                for k, j in enumerate(texts_index):
                    tmp[f'{text_index}'].iloc[j] = ". ".join(text_batch[k])
                    tmp[f'{text_index}_translated'].iloc[j] = True

            # Find out at what index we should start from.
            start_from = 0
            files = os.listdir(f'data/output/3_data_translate/{index_lang}_{to_lang}_{text_index}')
            for file in files:
                file_index = file.split('_')
                if len(file_index) > 0:
                    if file_index[0].isdigit():
                        if int(file_index[0]) > start_from:
                            start_from = int(file_index[0])

            last_index = 0
            texts_length = []
            texts_index = []
            texts = []

            # For each row do translate.
            if tmp.shape[0] - start_from - 1 > 0:
                for index, row in tqdm(islice(tmp.iterrows(), start_from, None), total=tmp.shape[0] - start_from):

                    # Stack up texts for multiple translations at once.
                    if not row[f'{text_index}_translated'] and row[f'{text_index}_lang'] == index_lang and len(row[f'{text_index}']) > 0:
                        tmp_texts = [t[:511] for t in row[f'{text_index}'].split('.') if len(t) > 1]
                        tmp_texts = tmp_texts[:16]
                        texts_length.append(len(tmp_texts))
                        texts_index.append(index)
                        for t in tmp_texts:
                            texts.append(t)

                    # If more than 2 texts, then translate.
                    if len(texts) > 16 and index != 0:
                        handle(texts, texts_index, texts_length)
                        texts_length = []
                        texts_index = []
                        texts = []

                    # Save file.
                    if index % 1000 == 0 and index != 0:
                        last_index = index
                        save_tmp = tmp[['requestId', f'{text_index}_translated', text_index]]
                        save_tmp.iloc[index - 1000:index + 1].to_csv(f'output/3_data_translate/{index_lang}_{to_lang}_{text_index}/{index}_{path_output}', index=False)

                # Translate remaining in texts-array
                handle(texts, texts_index, texts_length)
                save_tmp = tmp[['requestId', f'{text_index}_translated', text_index]]
                save_tmp.iloc[last_index:].to_csv(f'output/3_data_translate/{index_lang}_{to_lang}_{text_index}/{len(tmp) - 1}_{path_output}', index=False)
                    
        for index in ['subject', 'description']:
            translate(index)


    def run(self):

        if not os.path.isfile(PATH_OUTPUT_CHECKPOINT_TRANSLATE_DETECT):
            self.checkpoint_detect()
        else:
            print(f'Skip checkpoint_detect(). To rerun delete {PATH_OUTPUT_CHECKPOINT_TRANSLATE_DETECT}')
            self.df_detected = pd.read_csv(PATH_OUTPUT_CHECKPOINT_TRANSLATE_DETECT)

        self.checkpoint_translate(
            to_lang='en',
            from_lang='da',
            index_lang='da',
            path_output=PATH_OUTPUT_CHECKPOINT_TRANSLATE_EN
        )

DataTranslate(debug=False)