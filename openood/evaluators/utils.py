from openood.utils import Config
from .base_evaluator import BaseEvaluator
from .fsood_evaluator import FSOODEvaluator
from .ood_evaluator import OODEvaluator


def get_evaluator(config: Config):
    evaluators = {
        'base': BaseEvaluator,
        'ood': OODEvaluator,
        'fsood': FSOODEvaluator,
    }
    if config.evaluator.name not in evaluators:
        raise KeyError(f"Unsupported evaluator in minimal build: {config.evaluator.name}")
    return evaluators[config.evaluator.name](config)
