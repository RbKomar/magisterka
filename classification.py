import pathlib

from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import Model
from tensorflow.keras.layers import concatenate, Dense, MaxPooling2D, Flatten, Activation, Dropout
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report
import os
import numpy as np
from DAE import load_image, load_model
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.layers import Layer
import keras.layers as kl
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing


class SoftAttention(Layer):
    def __init__(self, ch, m, concat_with_x=False, aggregate=False, **kwargs):
        self.channels = int(ch)
        self.multiheads = m
        self.aggregate_channels = aggregate
        self.concat_input_with_scaled = concat_with_x

        super(SoftAttention, self).__init__(**kwargs)

    def build(self, input_shape):

        self.i_shape = input_shape

        kernel_shape_conv3d = (self.channels, 3, 3) + (1, self.multiheads)  # DHWC

        self.out_attention_maps_shape = input_shape[0:1] + (self.multiheads,) + input_shape[1:-1]

        if self.aggregate_channels == False:

            self.out_features_shape = input_shape[:-1] + (input_shape[-1] + (input_shape[-1] * self.multiheads),)
        else:
            if self.concat_input_with_scaled:
                self.out_features_shape = input_shape[:-1] + (input_shape[-1] * 2,)
            else:
                self.out_features_shape = input_shape

        self.kernel_conv3d = self.add_weight(shape=kernel_shape_conv3d,
                                             initializer='he_uniform',
                                             name='kernel_conv3d')
        self.bias_conv3d = self.add_weight(shape=(self.multiheads,),
                                           initializer='zeros',
                                           name='bias_conv3d')

        super(SoftAttention, self).build(input_shape)

    def call(self, x):

        exp_x = K.expand_dims(x, axis=-1)

        c3d = K.conv3d(exp_x,
                       kernel=self.kernel_conv3d,
                       strides=(1, 1, self.i_shape[-1]), padding='same', data_format='channels_last')
        conv3d = K.bias_add(c3d,
                            self.bias_conv3d)
        conv3d = kl.Activation('relu')(conv3d)

        conv3d = K.permute_dimensions(conv3d, pattern=(0, 4, 1, 2, 3))

        conv3d = K.squeeze(conv3d, axis=-1)
        conv3d = K.reshape(conv3d, shape=(-1, self.multiheads, self.i_shape[1] * self.i_shape[2]))

        softmax_alpha = K.softmax(conv3d, axis=-1)
        softmax_alpha = kl.Reshape(target_shape=(self.multiheads, self.i_shape[1], self.i_shape[2]))(softmax_alpha)

        if self.aggregate_channels == False:
            exp_softmax_alpha = K.expand_dims(softmax_alpha, axis=-1)
            exp_softmax_alpha = K.permute_dimensions(exp_softmax_alpha, pattern=(0, 2, 3, 1, 4))

            x_exp = K.expand_dims(x, axis=-2)

            u = kl.Multiply()([exp_softmax_alpha, x_exp])

            u = kl.Reshape(target_shape=(self.i_shape[1], self.i_shape[2], u.shape[-1] * u.shape[-2]))(u)

        else:
            exp_softmax_alpha = K.permute_dimensions(softmax_alpha, pattern=(0, 2, 3, 1))

            exp_softmax_alpha = K.sum(exp_softmax_alpha, axis=-1)

            exp_softmax_alpha = K.expand_dims(exp_softmax_alpha, axis=-1)

            u = kl.Multiply()([exp_softmax_alpha, x])

        if self.concat_input_with_scaled:
            o = kl.Concatenate(axis=-1)([u, x])
        else:
            o = u

        return [o, softmax_alpha]

    def compute_output_shape(self, input_shape):
        return [self.out_features_shape, self.out_attention_maps_shape]

    def get_config(self):
        return super(SoftAttention, self).get_config()


def create_dataset(img_folder, max=float('inf')):
    files = os.listdir(img_folder)
    img_data_array = []
    y = []
    # resize with proportional factor
    img_height = 183
    img_width = 244
    metadata = pd.read_csv("HAM10000_metadata.csv")
    cnt = 0
    autoencoder = load_model()
    for file in files:
        img = load_image(img_folder, file, img_height, img_width)
        decoded_img = autoencoder.decoder(autoencoder.encoder(img.reshape(1, 244, 183, 3)).numpy()).numpy()
        img_data_array.append(np.resize(tf.squeeze(decoded_img).numpy(), (299, 299, 3)))
        y.append(metadata[metadata["image_id"] == file.split('.')[0]]["dx"].values[0])
        cnt += 1
        if cnt > max:
            break
    return np.asarray(img_data_array), y


def load_data():
    IMGS_DIR = r"D:\STUDIA\IBM\SEM2\WK\PROJEKT"
    original_dir = pathlib.Path(os.path.join(IMGS_DIR, r"HAM10000"))
    x, y = create_dataset(original_dir, 1000)
    le = preprocessing.LabelEncoder()
    le.fit(["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"])
    y = le.transform(y)
    size = 0.8*len(x)
    x_train, x_test = x[::size], x[size::]
    y_train, y_test = y[::size], y[size::]
    return x_train, x_test, y_train, y_test


def create_model():
    irv2 = tf.keras.applications.InceptionResNetV2(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classifier_activation="softmax")
    # Excluding the last 28 layers of the model - author of SOTA
    conv = irv2.layers[-28].output
    attention_layer, map2 = SoftAttention(aggregate=True, m=16, concat_with_x=False, ch=int(conv.shape[-1]),
                                          name='soft_attention')(conv)
    attention_layer = (MaxPooling2D(pool_size=(2, 2), padding="same")(attention_layer))
    conv = (MaxPooling2D(pool_size=(2, 2), padding="same")(conv))

    conv = concatenate([conv, attention_layer])
    conv = Activation('relu')(conv)
    conv = Dropout(0.5)(conv)
    output = Flatten()(conv)
    output = Dense(7, activation='softmax')(output)
    model = Model(inputs=irv2.input, outputs=output)
    opt = tf.keras.optimizers.Adam(learning_rate=0.01, epsilon=0.1)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train_model():
    dir = r"D:\STUDIA\IBM\SEM2\WK\PROJEKT"
    original_dir = pathlib.Path(os.path.join(dir, r"HAM10000"))
    x, y = create_dataset(original_dir, 1000)
    le = preprocessing.LabelEncoder()
    y = to_categorical(le.fit_transform(y))

    model = create_model()
    # class weights given by SOTA author
    class_weights = {
        0: 1.0,  # akiec
        1: 1.0,  # bcc
        2: 1.0,  # bkl
        3: 1.0,  # df
        4: 5.0,  # mel
        5: 1.0,  # nv
        6: 1.0,  # vasc
    }
    callbacks = [
        ModelCheckpoint(filepath='IRV2+SA.hdf5', monitor='val_accuracy', save_best_only=True, save_weights_only=True),
        EarlyStopping(monitor='val_loss', mode='min', patience=2, min_delta=0.001)
    ]
    history = model.fit(x, y,
                        epochs=3,
                        verbose=2,
                        batch_size=32,
                        validation_split=0.2, callbacks=callbacks, class_weight=class_weights)
    model.save_weights("IRV2+SA.hdf5")

    return history


def test_model():
    _, x_test, _, y_test = load_data()

    model = create_model()
    model.load_weights("IRV2+SA.hdf5")
    predictions = model.predict(x_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_test = to_categorical(y_test)

    targetnames = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    report = classification_report(y_test, y_pred, target_names=targetnames)

    print("\nClassification Report:")
    print(report)
    report = classification_report(y_test, y_pred, target_names=targetnames)
    print("\nClassification Report:")
    print(report)
    print("Precision: " + str(precision_score(y_test, y_pred, average='weighted')))
    print("Recall: " + str(recall_score(y_test, y_pred, average='weighted')))
    print("Accuracy: " + str(accuracy_score(y_test, y_pred)))
    for i in range(7):
        r = roc_auc_score(y_test[:, i], predictions[:, i])
        print("The ROC AUC score of " + targetnames[i] + " is: " + str(r))


def main():
    train_model()
    test_model()


if __name__ == '__main__':
    main()
