"""
config_loader.py

Loads the model configuration based on the
specifications in model.toml.
"""

from dataclasses import dataclass
from typing import Dict

import tomli


@dataclass
class DatasetConfig:
    """
    Various configuration information related to the dataset.
    Be sure to add attributes to this class as you add attributes
    to model.toml.
    """
    batch_size: int
    image_width: int
    image_height: int
    seed: int
    validation_split: float
    test_split: float

    def __init__(self, config_data: Dict[str, any]) -> None:
        dataset_config_data = config_data["dataset"]
        for key, value in dataset_config_data.items():
            setattr(self, key, value)


@dataclass
class ModelConfig:
    """
    Various configuration information related to the model.
    Be sure to add attributes to this class as you add attributes
    to model.toml.
    """
    seed: int
    learning_rate: float
    num_epochs: int

    def __init__(self, config_data: Dict[str, any]) -> None:
        model_config_data = config_data["model"]
        for key, value in model_config_data.items():
            setattr(self, key, value)


@dataclass
class Config:
    """
    The sum of all the different parts of configuration.
    """
    dataset_config: DatasetConfig
    model_config: ModelConfig


class ConfigLoader:
    """
    Loads the configuration data from a file. This configuration
    is exposed to a user of this class by get_config().
    """

    def __init__(self, config_path: str = "model.toml") -> None:
        """
        Constructs a new instance of the ConfigLoader class.

        :param config_path: THe absolute path to the configuration file.
        """
        self._config_path = config_path
        with open(config_path, 'rb') as config_handle:
            self._raw_config_data = tomli.load(config_handle)
        self.config = Config(self._read_dataset_config(), self._read_model_config())

    def get_config(self) -> Config:
        """
        Obtains the configuration data.

        :return: The configuration data.
        """
        return self.config

    def _read_model_config(self) -> ModelConfig:
        """
        Read the model configuration into memory.

        :return: The model configuration data.
        """
        return ModelConfig(self._raw_config_data)

    def _read_dataset_config(self) -> DatasetConfig:
        """
        Read the dataset configuration into memory.

        :return: The dataset configuration data.
        """
        return DatasetConfig(self._raw_config_data)
