from torch.utils.data import DataLoader

from openood.utils import Config
from .sae_trainer import SAETrainer


def get_trainer(net, train_loader: DataLoader, val_loader: DataLoader, config: Config):
    trainers = {
        'sae': SAETrainer,
    }
    if config.trainer.name not in trainers:
        raise KeyError(f"Unsupported trainer in minimal build: {config.trainer.name}")
    return trainers[config.trainer.name](net, train_loader, config)
