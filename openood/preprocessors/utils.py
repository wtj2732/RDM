from openood.utils import Config
from .base_preprocessor import BasePreprocessor
from .test_preprocessor import TestStandardPreProcessor


def get_preprocessor(config: Config, split):
    if config.preprocessor.name != 'base':
        raise KeyError(f"Unsupported preprocessor in minimal build: {config.preprocessor.name}")
    if split == 'train':
        return BasePreprocessor(config)
    return TestStandardPreProcessor(config)
