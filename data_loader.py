"""
data_loader.py

Provides data loading functionality to the
rest of the application.
"""
import os
from typing import Optional

import tensorflow as tf
from dataclasses import dataclass

from config_loader import Config


@dataclass
class ModelData:
    """
    Class for keeping track of different pieces of data,
    include training, testing, and validation data.
    """
    train: tf.data.Dataset
    validation: tf.data.Dataset


class DataLoader:
    """
    Provides utilities for loading data
    that is stored on the disk into RAM.
    """

    def __init__(self, config: Config, directory: os.PathLike = "data/PetImages/") -> None:
        """
        Initializes a new instance of the DataLoader class.

        :param config: The dataset configuration information.
        :param directory: The directory that the image files are located.
        """
        self.directory = directory
        self.config = config

    def load_data(self, transformation: Optional[tf.keras.Sequential] = None) -> ModelData:
        """
        Loads the data from the directory into memory.

        The data has the following operations performed on it
        after being loaded from the disk (not necessarily in this order):

        1. Transformed - Various transformations are applied.
        2. Resized - The images are resized to a fixed size for
        easier consumption by ML models further up the pipeline.
        3. Shuffled - The images are shuffled.
        4. Split into train, test, and validation sets.

        For those of you who say don't give implementation details
        in function comments, to you I say, work better with your team.


        :param transformation: A transformation that can be applied to the dataset.

        :return: ModelData - A de-facto struct containing
                 various data that can be consumed by the Keras later.
        """
        training_image_dataset: tf.data.Dataset = tf.keras.preprocessing.image_dataset_from_directory(
            self.directory,
            labels="inferred",
            label_mode='binary',
            class_names=None,
            color_mode="rgb",
            batch_size=self.config.model_config.batch_size,
            image_size=(self.config.dataset_config.image_width, self.config.dataset_config.image_height),
            shuffle=True,
            seed=self.config.dataset_config.seed,
            validation_split=self.config.dataset_config.validation_split,
            subset="training",
            interpolation="bilinear")

        validation_image_dataset: tf.data.Dataset = tf.keras.preprocessing.image_dataset_from_directory(
            self.directory,
            labels="inferred",
            label_mode='binary',
            class_names=None,
            color_mode="rgb",
            batch_size=self.config.model_config.batch_size,
            image_size=(self.config.dataset_config.image_width, self.config.dataset_config.image_height),
            shuffle=True,
            seed=self.config.dataset_config.seed,
            validation_split=self.config.dataset_config.validation_split,
            subset="validation",
            interpolation="bilinear")

        # perform transformation on training data
        if transformation is not None:
            training_image_dataset.map(lambda x, y: (transformation(x), y))

        # prefetch some images
        training_image_dataset = training_image_dataset.prefetch(self.config.model_config.batch_size)
        validation_image_dataset = validation_image_dataset.prefetch(self.config.model_config.batch_size)
        return ModelData(training_image_dataset, validation_image_dataset)


def image_is_corrupt(image_path: os.PathLike):
    """
    Tests whether an image is corrupt.

    :param image_path: The image path.
    :return: Whether the image is corrupt.
    """
    try:
        img_bytes = tf.io.read_file(image_path)
        tf.io.decode_image(img_bytes)
    except tf.errors.InvalidArgumentError:
        return True

    return False
