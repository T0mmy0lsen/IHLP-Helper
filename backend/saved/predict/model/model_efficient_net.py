import cv2
import numpy as np
import keras
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical

import albumentations as albu

# Resizing the images as per EfficientNetB0 to size (224, 224)
from predict.model.prepare import Prepare
from predict.model.preprocess import Preprocess
from predict.model.shared import SharedDict

from predict.model.model_cnn import ModelCNN
from predict.model.wordembedding import WordEmbeddingLoader, WordEmbedding

height = 224
width = 224
channels = 3

n_classes = 100
input_shape = (height, width, channels)

epochs = 15
batch_size = 32


def get_labels(labels):
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    return le

def create_dataset():
    shared = SharedDict().revised()
    shared.word_embedding_dim = 224
    Preprocess(shared)
    Prepare(
        shared,
        category_type='responsible'
    ).fetch(
        top=100,
        categorical=False,
        lang='da',
    )

    WordEmbedding(shared)
    WordEmbeddingLoader(shared, dim=shared.word_embedding_dim, output_sequence_length=224)

    x_train = ModelCNN.bulk_vectorize(shared, shared.x_train)
    x_validate = ModelCNN.bulk_vectorize(shared, shared.x_validate)

    le = get_labels(np.unique(np.concatenate([shared.y_train, shared.y_validate])))

    y_train_labels = le.transform(shared.y_train)
    y_validate_labels = le.transform(shared.y_validate)

    n_classes = 100

    y_train = to_categorical(y_train_labels, n_classes)
    y_validate = to_categorical(y_validate_labels, n_classes)

    print(x_train.shape)
    print(x_validate.shape)

    print(y_train.shape)
    print(y_validate.shape)

    return x_train, y_train, x_validate, y_validate, shared


class DataGenerator(keras.utils.Sequence):
    def __init__(self, images, labels=None, mode='fit', batch_size=batch_size, dim=(height, width), channels=channels,
                 n_classes=n_classes, shuffle=True, augment=False, shared=None):

        # initializing the configuration of the generator
        self.shared = shared
        self.images = images
        self.labels = labels
        self.mode = mode
        self.batch_size = batch_size
        self.dim = dim
        self.channels = channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    # method to be called after every epoch
    def on_epoch_end(self):
        self.indexes = np.arange(self.images.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    # return numbers of steps in an epoch using samples and batch size
    def __len__(self):
        return int(np.floor(len(self.images) / self.batch_size))

    # this method is called with the batch number as an argument to obtain a given batch of data
    def __getitem__(self, index):
        # generate one batch of data
        # generate indexes of batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # generate mini-batch of X
        X = np.empty((self.batch_size, *self.dim, self.channels))

        for i, ID in enumerate(batch_indexes):
            # generate pre-processed image
            img = self.images[ID]
            img = [self.shared.embedding_matrix[j] for j in img] * 3
            img = np.array(img).reshape(224, 224, 3)
            # image rescaling
            img = img.astype(np.float32) + 1.
            X[i] = img

        # generate mini-batch of y
        if self.mode == 'fit':
            y = self.labels[batch_indexes]

            # augmentation on the training dataset
            if self.augment == True:
                X = self.__augment_batch(X)
            return X, y

        elif self.mode == 'predict':
            return X

        else:
            raise AttributeError("The mode should be set to either 'fit' or 'predict'.")

    # augmentation for one image
    def __random_transform(self, img):
        composition = albu.Compose([albu.HorizontalFlip(p=0.5),
                                    albu.VerticalFlip(p=0.5),
                                    albu.GridDistortion(p=0.2),
                                    albu.ElasticTransform(p=0.2)])
        return composition(image=img)['image']

    # augmentation for batch of images
    def __augment_batch(self, img_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i] = self.__random_transform(img_batch[i])
        return img_batch

# X_train_data = np.random.randint(255, size=(100, 3, 5, 5))
# y_train_data = np.random.randint(1, size=(100, 100, 1))
# X_val_data = np.random.randint(255, size=(100, 3, 5, 5))
# y_val_data = np.random.randint(1, size=(100, 100, 1))

x_train, y_train, x_val, y_val, shared = create_dataset()

train_data_generator = DataGenerator(x_train, y_train, augment=True, shared=shared)
valid_data_generator = DataGenerator(x_val, y_val, augment=False, shared=shared)

import efficientnet.keras as efn

efnb0 = efn.EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape, classes=n_classes)

model = Sequential()
model.add(efnb0)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))

model.summary()

optimizer = Adam(lr=0.0001)

# Early stopping to monitor the validation loss and avoid overfitting
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)

# Reducing learning rate on plateau
rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=1)

# Model compiling
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model_history = model.fit_generator(train_data_generator,
                                    validation_data=valid_data_generator,
                                    callbacks=[early_stop, rlrop],
                                    verbose=1,
                                    epochs=epochs)

# Saving the trained model weights as data file in .h5 format
model.save_weights("cifar_efficientnetb0_weights.h5")