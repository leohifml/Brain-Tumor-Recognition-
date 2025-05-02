import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, ZeroPadding2D, MaxPooling2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pickle
import os
from data_loading_eda import images, labels, class_names


def image_modifications(image):
    """Applies preprocessing to the image (grayscale, Gaussian blur)."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return image

def preprocess_images(images):
    """Applies image modifications to a set of images and reshapes them."""

    images = np.array([image_modifications(image) for image in images])
    images = images.reshape((images.shape[0], 128, 128, 1))  # Assuming 128x128 grayscale
    return images

def identity_block(x, f, filters, training=True):
    """Identity block for ResNet."""

    F1, F2 = filters
    x_shortcut = x

    x = Conv2D(filters=F1, kernel_size=(f, f), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)

    return x

def convolutional_block(x, f, filters, s=2, training=True):
    """Convolutional block for ResNet (with shortcut)."""

    F1, F2 = filters
    x_shortcut = x

    x = Conv2D(filters=F1, kernel_size=(f, f), strides=(s, s), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)

    x_shortcut = Conv2D(filters=F2, kernel_size=(1, 1), strides=(s, s), padding='same')(x_shortcut)
    x_shortcut = BatchNormalization()(x_shortcut)

    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)

    return x

def ResNet18(input_shape=(224, 224, 3), classes_len=int):
    """ResNet18 model architecture."""

    x_input = Input(input_shape)
    x = ZeroPadding2D((3, 3))(x_input)

    x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = convolutional_block(x, f=3, filters=[64, 64], s=1)
    x = identity_block(x, 3, [64, 64])

    x = convolutional_block(x, f=3, filters=[128, 128], s=2)
    x = identity_block(x, 3, [128, 128])

    x = convolutional_block(x, f=3, filters=[256, 256], s=2)
    x = identity_block(x, 3, [256, 256])

    x = convolutional_block(x, f=3, filters=[512, 512], s=2)
    x = identity_block(x, 3, [512, 512])

    x = AveragePooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(classes_len, activation='softmax')(x)

    model = Model(inputs=x_input, outputs=x, name='ResNet18')
    return model

def train_model(model, images, labels, epochs=30, batch_size=32, validation_split=0.1):
    """Trains the given model."""

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=0.000001)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(x=images, y=labels, batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                        callbacks=[early_stop, reduce_lr])
    return history, model  # Return the trained model

def plot_training_history(history):
    """Plots training and validation accuracy and loss."""

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Usage
    train_img = preprocess_images(images)  # Apply preprocessing
    model1 = ResNet18(input_shape=(128, 128, 1), classes_len=len(class_names))
    history, trained_model = train_model(model1, train_img, labels)
    plot_training_history(history)