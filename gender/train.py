import pandas as pd
import logging
import argparse
import os
from keras.applications.vgg16 import VGG16
from keras.layers import Dense
from keras import optimizers
import numpy as np
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from PIL import Image
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
import keras


def main():
    img_width = 224
    img_height = 224
    batch_size = 40


    datagen = ImageDataGenerator(rescale=1. / 255)


    train_generator = datagen.flow_from_directory(
        './datas/train',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

    datagen2 = ImageDataGenerator(rescale=1. / 255)

    validation_generator = datagen2.flow_from_directory(
        './datas/validation',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)


    vgg = VGG16(weights='imagenet')
    x = vgg.get_layer('fc2').output
    prediction = Dense(2, activation='softmax', name='predictions')(x)
    model = Model(input=vgg.input, outputs=prediction)

    sgd = SGD(lr=0.0005, momentum=0.9, nesterov=True, decay=0.0001)
    model.compile(optimizer=sgd, loss="categorical_crossentropy",
                  metrics=['accuracy'])

    model.count_params()
    model.summary()

    callbacks = [
                 ModelCheckpoint("./checkpoints/vgg_sgd_pre2_weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="auto")
                 ]

    hist = model.fit_generator(train_generator, steps_per_epoch=1000, epochs=40,
                         validation_data=validation_generator, validation_steps=250, callbacks=callbacks)

    model.save_weights('my_model_weights2.h5')
    # logging.debug("Saving weights...")
    # model.save_weights(os.path.join("models", "WRN_{}_{}.h5".format(depth, k)), overwrite=True)
    # pd.DataFrame(hist.history).to_hdf(os.path.join("models", "history_{}_{}.h5".format(depth, k)), "history")


if __name__ == '__main__':
    main()
