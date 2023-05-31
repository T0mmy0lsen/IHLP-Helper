
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import recall_score, f1_score, precision_score, classification_report
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt

def sandbox(nrows=5000, LABEL='placement'):

    loaded_predictions = np.load(f'validate_{LABEL}_predictions.npy')
    df = pd.read_csv(f'data/cached_validate_{LABEL}.csv')
    df = df[-nrows:]
    df['logs'] = np.max(loaded_predictions, axis=-1)
    print(len(df[df.logs >= 9.7]) / nrows)
    print(len(df[df.logs >= 10.7]) / nrows)

    probabilities = tf.nn.softmax(loaded_predictions, axis=-1)
    predicted_classes = tf.argmax(probabilities, axis=-1)

    df['probabilities'] = probabilities.numpy().tolist()
    df['predicted_classes'] = predicted_classes.numpy()

    # Calculate the accuracy by comparing the predicted classes to the true labels
    true_labels = df.label.values
    accuracy = np.mean(df['predicted_classes'] == true_labels)

    print(f"Accuracy: {accuracy}")

    for log_threshold in tqdm(np.arange(0.0, 12, 0.1)):

        tmp = df.copy()
        tmp = tmp[tmp.logs >= log_threshold]

        # Calculate the accuracy by comparing the predicted classes to the true labels
        true_labels = tmp.label.values
        accuracy = np.mean(tmp['predicted_classes'] == true_labels)

        if accuracy > 0.9:
            print(f"Log threshold: {log_threshold}, Accuracy: {accuracy}")


def run(nrows=5000, LABEL='placement', MODEL='Placement'):

    model = TFAutoModelForSequenceClassification.from_pretrained(f'data/models/IHLP-XLM-RoBERTa-{MODEL}')
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    df = pd.read_csv(f'data/cached_test_html_tags_{LABEL}.csv')
    # df = df[-nrows:]

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

    model.compile(metrics=['acc'])
    predictions = model.predict(tokenized_text)
    predicted_log_values = predictions[0]

    np.save(f'validate_{LABEL}_predictions.npy', predicted_log_values)  # Save predictions to a file

    df['logs'] = np.max(predicted_log_values, axis=-1)
    probabilities = tf.nn.softmax(predicted_log_values, axis=-1)
    predicted_classes = tf.argmax(probabilities, axis=-1)

    df['probabilities'] = probabilities.numpy().tolist()
    df['predicted_classes'] = predicted_classes.numpy()

    # Calculate the accuracy by comparing the predicted classes to the true labels
    true_labels = df.label.values
    accuracy = np.mean(df['predicted_classes'] == true_labels)

    print(f"Accuracy: {accuracy}")

    precision = precision_score(true_labels, df['predicted_classes'], average='weighted')
    recall = recall_score(true_labels, df['predicted_classes'], average='weighted')
    f1 = f1_score(true_labels, df['predicted_classes'], average='weighted')

    predicted_classes = np.array(df['predicted_classes'])

    # Calculate the distribution of predicted classes
    unique, counts = np.unique(predicted_classes, return_counts=True)

    # Print the counts
    for label, count in zip(unique, counts):
        print(f"Count of predictions for class {label}: {count}")

    # Plot the counts
    plt.figure(figsize=(10, 6))
    plt.bar(unique, counts)
    plt.xlabel('Classes')
    plt.ylabel('Count of predictions')
    plt.title('Distribution of Predicted Classes')

    # Save the plot as a .png file
    plt.savefig('predicted_class_distribution.png')

    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 score: {f1}")

    """
        for log_threshold in tqdm(np.arange(0.0, 10.2, 0.1)):

        tmp = df.copy()
        tmp = tmp[tmp.logs >= log_threshold]

        # Calculate the accuracy by comparing the predicted classes to the true labels
        true_labels = tmp.label.values
        accuracy = np.mean(tmp['predicted_classes'] == true_labels)

        # Calculate precision, recall, and F1 score
        precision = precision_score(true_labels, tmp['predicted_classes'], average='weighted')
        recall = recall_score(true_labels, tmp['predicted_classes'], average='weighted')
        f1 = f1_score(true_labels, tmp['predicted_classes'], average='weighted')

        if accuracy > 0.9:
            print(f"Log threshold: {log_threshold}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 score: {f1}")
    """


run(LABEL='responsible', MODEL='Responsible', nrows=10000)
# sandbox(LABEL='responsible', nrows=10000)