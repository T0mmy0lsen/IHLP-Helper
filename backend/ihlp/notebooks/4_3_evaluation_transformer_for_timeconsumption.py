
import numpy as np
import pandas as pd

from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

MODEL = 'Time-Consumption'

model = TFAutoModelForSequenceClassification.from_pretrained(f'data/models/IHLP-XLM-RoBERTa-{MODEL}')
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

df = pd.read_csv(f'data/cached_html_tags_validate_timeconsumption.csv')

print(df.head())


def tokenize_texts(sentences, max_length=512, padding='max_length'):
    return tokenizer(
        sentences,
        truncation=True,
        padding=padding,
        max_length=max_length,
        return_tensors="tf"
    )

tokenized_text = dict(tokenize_texts(list(df['text'].values)))
true_values = df.label.values

model.compile(metrics=['mae'])

output = model.predict(tokenized_text)
predictions = output.logits[:, 0]
np.save('validate_timeconsumption_predictions.npy', predictions)  # Save predictions to a file

# Load saved predictions
loaded_predictions = np.load('validate_timeconsumption_predictions.npy')

errors = np.abs(loaded_predictions - true_values)

# Calculate mean absolute error (MAE)
mae = np.mean(errors)
print(f"Mean Absolute Error (MAE): {mae}")

below_threshold_indices_1 = np.where(errors <= 1.0)[0]
below_threshold_indices_2 = np.where(errors <= 2.0)[0]

filtered_predictions_1 = loaded_predictions[below_threshold_indices_1]
filtered_true_values_1 = true_values[below_threshold_indices_1]

filtered_predictions_2 = loaded_predictions[below_threshold_indices_2]
filtered_true_values_2 = true_values[below_threshold_indices_2]

percentage_below_threshold_1 = 100 * len(loaded_predictions) / len(filtered_predictions_1)
percentage_below_threshold_2 = 100 * len(loaded_predictions) / len(filtered_predictions_2)

print(f"Percentage of predictions with error below 1.0: {percentage_below_threshold_1}%")
print(f"Percentage of predictions with error below 2.0: {percentage_below_threshold_2}%")