from openood.utils import Config
from .base_postprocessor import BasePostprocessor
from .gmm_postprocessor import GMMPostprocessor


def get_postprocessor(config: Config):
    postprocessors = {
        'msp': BasePostprocessor,
        'gmm': GMMPostprocessor,
    }
    if config.postprocessor.name not in postprocessors:
        raise KeyError(f"Unsupported postprocessor in minimal build: {config.postprocessor.name}")
    return postprocessors[config.postprocessor.name](config)
