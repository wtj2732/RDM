from openood.utils import Config
from .base_recorder import BaseRecorder


def get_recorder(config: Config):
    recorders = {
        'base': BaseRecorder,
    }
    if config.recorder.name not in recorders:
        raise KeyError(f"Unsupported recorder in minimal build: {config.recorder.name}")
    return recorders[config.recorder.name](config)
