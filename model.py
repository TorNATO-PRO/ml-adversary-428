"""
model.py

This is where the actual task of binary classification
happens.
"""
import os
from typing import Any, Optional

import tensorflow as tf

from config_loader import Config


class ConvolutionalBlock(tf.keras.Model):
    """
    A convolutional block - used as a building block for the convolutional
    neural network that we are building.
    """

    def __init__(self, num_filters: int, kernel_size: int, pool_size: int, activation: tf.keras.activations) -> None:
        """
        Initializes a new instance of the ConvolutionalBlock.

        :param num_filters: The number of filters.
        :param kernel_size: The size of the kernel.
        :param pool_size: The pool size.
        :param activation: The activation function to use.
        """
        super(ConvolutionalBlock, self).__init__()
        self.conv_2d = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size)
        self.max_pool_2d = tf.keras.layers.MaxPool2D(pool_size=pool_size)
        self.batch_normalization = tf.keras.layers.BatchNormalization()
        self.activation = activation

    def call(self, inputs: Any, training: bool = False, mask: Any = None) -> None:
        """
        What happens when the class is "called" by Tensorflow.

        :param inputs: The inputs to the layer.
        :param training: Whether it is being used on training data.
        :param mask: The mask (I have no idea what it is, just implemented the interface).
        """
        x = self.conv_2d(inputs)
        x = self.max_pool_2d(x)
        x = self.batch_normalization(x, training=training)
        x = self.activation(x)
        return x


class Model:
    def __init__(self, config: Config, model_path: Optional[os.PathLike] = None):
        """
        Initializes a new instance of the model class.

        We are using a similar architecture as found here, with a few differences:
        https://keras.io/examples/vision/3D_image_classification/

        No way could I pull this off without some sort of reference.

        :param config: The configuration parameters.
        :param model_path: The absolute path to a saved model.
        """
        self.config = config
        self.model = tf.keras.Sequential([
                tf.keras.layers.Input(
                    (self.config.dataset_config.image_width, self.config.dataset_config.image_height, 3)),
                ConvolutionalBlock(32, 3, 2, tf.keras.activations.relu),
                ConvolutionalBlock(32, 3, 2, tf.keras.activations.relu),
                ConvolutionalBlock(64, 3, 2, tf.keras.activations.relu),
                ConvolutionalBlock(64, 3, 2, tf.keras.activations.relu),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(units=1024, activation=tf.keras.activations.relu),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)]
            )
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.model_config.learning_rate),
            metrics=["accuracy"],
        )

        if model_path:
            self._load_model(model_path)

    def fit_model(self, training_data: tf.data.Dataset, validation_data: tf.data.Dataset, save_directory: Optional[os.PathLike] = None) -> tf.keras.callbacks.History:
        """
        Fits the model on training data. We are using an early stopping callback
        to stop training if necessary.

        :param training_data: The training data that is being used.
        :param validation_data: The validation data that is being used.
        :param save_directory: The directory to save the model at.
        """
        checkpoint = None
        if save_directory is not None:
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                save_directory, save_best_only=True
            )

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)]
        if checkpoint is not None:
            callbacks += [checkpoint]

        history = self.model.fit(
            training_data,
            validation_data=validation_data,
            epochs=self.config.model_config.num_epochs,
            callbacks=callbacks,
            batch_size=self.config.model_config.batch_size,
            validation_batch_size=self.config.model_config.batch_size
        )
        return history

    def make_predictions(self, data) -> None:
        """
        Makes predictions on given data.

        :param data: The data to make predictions on.
        """
        return self.model.predict(data)

    def _load_model(self, path: os.PathLike) -> None:
        """
        Loads a model from an absolute path.

        :param path: The path to load the model from.
        """
        if not os.path.exists(path):
            return

        self.model.load_weights(path)
