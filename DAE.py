import pathlib

import numpy as np
import tensorflow as tf
import os
from matplotlib import pyplot as plt
import cv2


class Denoise(tf.keras.models.Model):
    def __init__(self):
        super(Denoise, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(244, 183, 3)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(16, kernel_size=3, activation='relu', padding='same'),
            tf.keras.layers.Conv2DTranspose(32, kernel_size=3, activation='relu', padding='same'),
            tf.keras.layers.Conv2DTranspose(64, kernel_size=3, activation='relu', padding='same'),
            tf.keras.layers.Conv2DTranspose(128, kernel_size=3, activation='relu', padding='same'),
            tf.keras.layers.Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def load_image(img_folder, file, img_height, img_width):
    image_path = os.path.join(img_folder, file)
    image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_height, img_width), interpolation=cv2.INTER_AREA)
    image = np.array(image)
    image = image.astype('float32')
    image /= 255
    return image


def create_dataset(img_folder, y=False):
    files = os.listdir(img_folder)
    img_data_array = []
    # resize with proportional factor
    img_height = 183
    img_width = 244

    for file in files:
        if y:
            for _ in range(4):
                img_data_array.append(load_image(img_folder, file, img_height, img_width))
        img_data_array.append(load_image(img_folder, file, img_height, img_width))
    return img_data_array


def load_data():
    IMGS_DIR = r"D:\STUDIA\IBM\SEM2\WK\PROJEKT\imgs"
    skin_dir = pathlib.Path(os.path.join(IMGS_DIR, r"without_hairs"))
    hairy_skin_dir = pathlib.Path(os.path.join(IMGS_DIR, r"with_hairs"))
    X = create_dataset(hairy_skin_dir)
    y = create_dataset(skin_dir, y=True)
    return X, y


def main():
    x, y = load_data()
    x = np.asarray(x)
    y = np.asarray(y)

    autoencoder = Denoise()
    autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    autoencoder.fit(x, y,
                    epochs=100,
                    shuffle=True,
                    batch_size=32,
                    validation_split=0.2)
    encoded_imgs = autoencoder.encoder(y).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original + noise
        ax = plt.subplot(2, n, i + 1)
        plt.title("original + noise")
        plt.imshow(tf.squeeze(x[i]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        bx = plt.subplot(2, n, i + n + 1)
        plt.title("reconstructed")
        plt.imshow(tf.squeeze(decoded_imgs[i]))
        plt.gray()
        bx.get_xaxis().set_visible(False)
        bx.get_yaxis().set_visible(False)
    plt.show()


if __name__ == '__main__':
    main()
