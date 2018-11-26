from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import regularizers
from keras import layers
from keras import optimizers
import os
from time import time

def build_model_64x64(experiment_name, epochs, batch_size, lamda, learning_rate, optimizer_type, path_to_64):
    img_width = 64
    img_height = 64
    input_shape = (img_width, img_height, 3)

    train_data_dir = path_to_64 + '/train'
    validation_data_dir = path_to_64 + '/validate'

    nb_train_samples = sum(len(files) for _, _, files in os.walk(train_data_dir))
    nb_validation_samples = sum(len(files) for _, _, files in os.walk(validation_data_dir))

    callbacks = [TensorBoard(log_dir="logs/{}".format(time())),
                 EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=1, mode='auto', baseline=None,
                               restore_best_weights=False),
                 ModelCheckpoint(experiment_name + '.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=0,
                                 save_best_only=True, save_weights_only=True,
                                 mode='auto', period=1)]

    # ####DEFECT MODEL START#####
    model = Sequential()
    # Add a dropout layer for input layer
    model.add(layers.Dropout(0.2, input_shape=input_shape))
    # Convolution layer: 32 filters, kernal size 3 x 3, L2 regularization
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, strides=2, kernel_regularizer=regularizers.l2(lamda)))
    model.add(Activation('relu'))
    # Pooling layer: subsampling 2 x 2, stride 2
    model.add(MaxPooling2D(pool_size=(2, 2), strides=1))
    # Convolution layer: 64 filters, kernal size 3 x 3, L2 regularization
    model.add(Conv2D(64, (3, 3), input_shape=input_shape, kernel_regularizer=regularizers.l2(lamda)))
    model.add(Activation('relu'))
    # Convolution layer: 64 filters, kernal size 3 x 3, L2 regularization
    model.add(Conv2D(64, (3, 3), input_shape=input_shape, kernel_regularizer=regularizers.l2(lamda)))
    model.add(Activation('relu'))
    # Pooling layer: subsampling 2 x 2, stride 2
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    # Convolution layer: 128 filters, kernal size 3 x 3, L2 regularization
    model.add(Conv2D(128, (3, 3), input_shape=input_shape, kernel_regularizer=regularizers.l2(lamda)))
    model.add(Activation('relu'))
    # Convolution layer: 128 filters, kernal size 3 x 3, L2 regularization
    model.add(Conv2D(128, (3, 3), input_shape=input_shape, kernel_regularizer=regularizers.l2(lamda)))
    model.add(Activation('relu'))
    # Pooling layer: subsampling 2 x 2, stride 2
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    # Fully connected layer: 1024 Activation Units
    model.add(layers.Dense(units=1024, activation='relu'))
    # Dropout layer probability 0.5
    model.add(layers.Dropout(0.5))
    # Fully connected layer: 1024 Activation Units
    model.add(layers.Dense(units=1024, activation='relu'))
    # Dropout layer probability 0.5
    model.add(layers.Dropout(0.5))
    # Add fully connected layer with a sigmoid activation function
    model.add(layers.Dense(units=1, activation='sigmoid'))  # org
    print(model.summary())
    #   #####DEFECT MODEL END######

    if optimizer_type == 'RMSprop':
        optimizer = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
    elif optimizer_type == 'Adam':
        optimizer = optimizers.adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Datagenerators
    train_datagen = ImageDataGenerator(rotation_range=5, width_shift_range=0.1, height_shift_range=0.1,
                                       rescale=1. / 255,
                                       zoom_range=0.1, horizontal_flip=False, fill_mode='nearest')

    valid_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height),
                                                        batch_size=batch_size, class_mode='binary', shuffle=True)

    validation_generator = valid_datagen.flow_from_directory(validation_data_dir, target_size=(img_width, img_height),
                                                             batch_size=batch_size, class_mode='binary', shuffle=True)

    model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs,
                        validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size,
                        verbose=1, callbacks=callbacks)

    return model
