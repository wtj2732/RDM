from openood.utils import Config

from .finetune_pipeline import FinetunePipeline
from .test_ood_pipeline import TestOODPipeline


def get_pipeline(config: Config):
    pipelines = {
        'finetune': FinetunePipeline,
        'test_ood': TestOODPipeline,
    }
    if config.pipeline.name not in pipelines:
        raise KeyError(f"Unsupported pipeline in minimal build: {config.pipeline.name}")
    return pipelines[config.pipeline.name](config)
