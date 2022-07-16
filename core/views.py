from django.shortcuts import render
from django.contrib.auth.models import User
from rest_framework.views import APIView
from rest_framework.response import Response
import json

# ===========================================================================================================
import numpy as np
import sys
from keras.layers import Input
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, MaxPooling2D
from keras.layers import Dense, Flatten, SpatialDropout2D
from keras.layers.merging import concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import History
from keras.models import load_model
from sklearn.metrics import confusion_matrix

image_size = (64, 64)


def buildNet(num_classes):
    """
    Function to build 4 layer NN with 2 Conv layers, 1 MaxPool layer,
    1 GlobalMaxPool layer and 2 Dense layers

    Parameters
    ----------
    num_classes: int
                 Number of classes in training data
    Returns
    -------
    Neural Network created
    """
    model1 = Sequential()
    model1.add(Convolution2D(32, (3, 3), input_shape=(image_size[0], image_size[1], 3), activation='relu'))
    model1.add(MaxPooling2D(pool_size=(2, 2)))
    model1.add(Convolution2D(64, (3, 3), activation='relu'))
    model1.add(GlobalAveragePooling2D())

    model1.add(Dense(128, activation='relu'))
    model1.add(Dense(1, activation='sigmoid'))
    model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model1.summary())
    return model1


def trainNet(training_set, validation_set):
    """
    Function to train Neural Network Created, save it as hd5 and plot the various parameters.

    Arguments
    ---------
    training_set:   ImageDataGenerator object
                    Training set with labels.
    validation_set: ImageDataGenerator object
                    Validation set with labels.

    Returns
    -------
    history: dictionary
             History of training and validation of model.
    """
    num_classes = 1  # y_train.shape[1]
    model = buildNet(num_classes)
    history = History()
    callbacks = [EarlyStopping(monitor='val_loss', patience=5),
                 ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True), history]

    history = model.fit(training_set,
                        steps_per_epoch=8000 / 32,
                        epochs=1,
                        validation_data=validation_set,
                        validation_steps=64,
                        use_multiprocessing=True,
                        workers=8)
    model.save('model.hd5')

    # model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=100,callbacks=callbacks,verbose=1)
    return history


def result():
    """
    Function to predict if the retina image has diabetic retinopathy or not.

    Parameters
    ----------
    None

    Returns
    -------
    y_pred: bool
            Whether or not the retina has diabetic retinopathy.
    percent_chance: float
            Percentage of chance the retina image has diabetic retinopathy.
    """

    mod = load_model('model.hd5')

    test_gen = ImageDataGenerator(rescale=1. / 255)

    test_data = test_gen.flow_from_directory('e:/SamenPFE/menrel/gaussian_filtered_images',
                                             target_size=(64, 64),
                                             batch_size=48,
                                             class_mode='binary', shuffle=False)
    predicted = mod.predict(test_data)

    y_pred = predicted[0][0] > 0.4
    print(predicted)
    percent_chance = round(predicted[0][0] * 100, 2)

    return y_pred, percent_chance


def predic(path):
    global mod
    test_gen = ImageDataGenerator(rescale=1. / 255)

    test_data = test_gen.flow_from_directory(path,
                                             target_size=(64, 64),
                                             batch_size=48,
                                             class_mode='binary', shuffle=False)
    predicted = mod.predict(test_data)

    y_pred = predicted[0][0] > 0.4
    print(predicted)
    percent_chance = round(predicted[0][0] * 100, 2)

    return [y_pred, percent_chance]


def main(path):
    train_datagen = ImageDataGenerator(rescale=1. / 255)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory('e:/SamenPFE/menrel/gaussian_filtered_images',
                                                     target_size=image_size,
                                                     batch_size=48)
    # class_mode = 'binary')
    # training_set

    validation_set = test_datagen.flow_from_directory('e:/SamenPFE/menrel/gaussian_filtered_images',
                                                      target_size=image_size,
                                                      batch_size=48, shuffle=False)
    # class_mode = 'binary', shuffle=False)

    history = trainNet(training_set=training_set, validation_set=validation_set)

    mod = load_model('model.hd5')

    return predic(path)[0]


# ===========================================================================================================

# restframework config
from rest_framework import routers, serializers, viewsets


# ViewSets define the view behavior.
# Serializers define the API representation.

class TestViewset(APIView):
    def post(self, request):
        print("processus de detection enclenché\n")
        print(f"image: {request.data['file']}")
        print("||\n||\n||\n||\n||\n||\n\\/")
        result = {
            "status": main(request.data['file'])
        }
        return Response(json.dumps(result))


# Create your views here
def Test(request):
    return render(request.POST['test'])
