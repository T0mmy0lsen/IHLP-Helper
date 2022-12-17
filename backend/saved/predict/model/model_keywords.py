import time
from itertools import repeat

import numpy as np
from nltk import FreqDist
from sklearn import preprocessing
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, top_k_accuracy_score
from tqdm import tqdm


class ModelKeywords:

    def __init__(self, shared):
        self.shared = shared
        self.run()

    def run(self):

        def get_data_validate():
            data_dict = {}
            for i in self.shared.categories:
                data_dict[i] = []
            for idx, el in enumerate(self.shared.x_validate):
                data_dict[self.shared.y_validate[idx][:100]].append(el)
            return data_dict

        def get_data_train():
            data_dict = {}
            for i in self.shared.categories:
                data_dict[i] = []
            for idx, el in enumerate(self.shared.x_train):
                data_dict[self.shared.y_train[idx][:100]].append(el)
            return data_dict

        def get_labels(labels):
            le = preprocessing.LabelEncoder()
            le.fit(labels)
            return le

        def create_vectorizers(train_dict):
            topic_list = list(train_dict.keys())
            vect_dict = {}
            for topic in topic_list:
                text_array = train_dict[topic]
                text = " ".join(text_array)
                word_list = text.split(" ")
                freq_dist = FreqDist(word_list)
                # vect_dict[topic] = CountVectorizer(max_df=.9, vocabulary=[e[0] for e in freq_dist.most_common(200)])
                vect_dict[topic] = CountVectorizer(max_df=.9, vocabulary=[e[0] for e in freq_dist.most_common(1000)])
            return vect_dict

        def create_dataset(data_dict, vect_dict, le, sum_list):
            data_matrix = []
            gold_labels = []
            for category_index, category in enumerate(tqdm(vect_dict.keys())):
                labels = [le.transform([category])] * len(data_dict[category])
                gold_labels = gold_labels + labels
                for data_index, data in enumerate(data_dict[category]):
                    data_matrix.append([e[category_index][data_index] for e in sum_list])
            x = np.array(data_matrix)
            y = np.array(gold_labels)
            return x, y

        def create_dataset_sum_list(data_dict, vect_dict):
            sum_list = []
            for vector_index, vector in enumerate(tqdm(vect_dict.keys())):
                sum_list.append([])
                vectorizer = vect_dict[vector]
                for data_index, data in enumerate(vect_dict.keys()):
                    sum_list[vector_index].append([])
                    category_matrix = vectorizer.transform(data_dict[data]).todense()
                    for i, m in enumerate(category_matrix):
                        total = sum(category_matrix[i].tolist()[0])
                        sum_list[vector_index][data_index].append(total)
            return sum_list


        def classify(vector, le):
            result = sorted(range(len(vector)), key=lambda i: vector[i])[-3:]
            label = [1 if i in result else 0 for i in list(range(len(vector)))]
            return label

        def evaluate(x, y, le):
            y_pred = np.array(list(map(classify, x, repeat(le))))
            y_validation = y.flatten()
            print("[Keywords] Actual Score:", top_k_accuracy_score(y_validation, y_pred, k=3))
            print("")
            # print(classification_report(y, y_pred))

        train_dict = get_data_train()
        validate_dict = get_data_validate()

        print("[Keywords] Create Vectorizer")
        vect_dict = create_vectorizers(train_dict)

        print("[Keywords] Create Data")
        labels = get_labels(list(vect_dict.keys()))
        sum_list = create_dataset_sum_list(validate_dict, vect_dict)
        x, y = create_dataset(validate_dict, vect_dict, labels, sum_list)

        print("[Keywords] Predict")
        evaluate(x, y, labels)
