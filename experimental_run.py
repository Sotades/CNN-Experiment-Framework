from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras import optimizers
import datetime
import os
from time import time
from build_model128x128 import build_model_128x128
from build_model64x64 import build_model_64x64
from save_model_and_weights import save_model_and_weights

def experimental_run(experiment_name, epochs, image_size, lamda, minibatch_size, learning_rate, optimizer, path_to_128=None, path_to_64=None):

    if image_size == 128:
        model = build_model_128x128(experiment_name=experiment_name,
                                    epochs=epochs,
                                    batch_size=minibatch_size,
                                    lamda=lamda,
                                    learning_rate=learning_rate,
                                    optimizer_type=optimizer,
                                    path_to_128=path_to_128)
    elif image_size == 64:
        model = build_model_64x64(experiment_name=experiment_name,
                                  epochs=epochs,
                                  batch_size=minibatch_size,
                                  lamda=lamda,
                                  learning_rate=learning_rate,
                                  optimizer_type=optimizer,
                                  path_to_64=path_to_64)

    save_model_and_weights(experiment_name, model)

    return model
