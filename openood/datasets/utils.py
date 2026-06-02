import torch
from torch.utils.data import DataLoader

from openood.preprocessors.test_preprocessor import TestStandardPreProcessor
from openood.preprocessors.utils import get_preprocessor
from openood.utils.config import Config
from .imglist_dataset import ImglistDataset


def _build_imglist_dataset(name, split_config, dataset_config, preprocessor, data_aux_preprocessor):
    return ImglistDataset(
        name=name,
        imglist_pth=split_config.imglist_pth,
        data_dir=split_config.data_dir,
        num_classes=dataset_config.num_classes,
        preprocessor=preprocessor,
        data_aux_preprocessor=data_aux_preprocessor,
    )


def get_dataloader(config: Config):
    dataset_config = config.dataset
    dataloader_dict = {}
    for split in dataset_config.split_names:
        split_config = dataset_config[split]
        if split_config.dataset_class != 'ImglistDataset':
            raise KeyError(f"Unsupported dataset_class in minimal build: {split_config.dataset_class}")
        preprocessor = get_preprocessor(config, split)
        data_aux_preprocessor = TestStandardPreProcessor(config)
        dataset = _build_imglist_dataset(
            dataset_config.name + '_' + split,
            split_config,
            dataset_config,
            preprocessor,
            data_aux_preprocessor,
        )
        sampler = None
        if dataset_config.num_gpus * dataset_config.num_machines > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            split_config.shuffle = False
        dataloader_dict[split] = DataLoader(
            dataset,
            batch_size=split_config.batch_size,
            shuffle=split_config.shuffle,
            num_workers=dataset_config.num_workers,
            sampler=sampler,
        )
    return dataloader_dict


def get_ood_dataloader(config: Config):
    ood_config = config.ood_dataset
    if ood_config.dataset_class != 'ImglistDataset':
        raise KeyError(f"Unsupported ood dataset_class in minimal build: {ood_config.dataset_class}")
    dataloader_dict = {}
    for split in ood_config.split_names:
        split_config = ood_config[split]
        preprocessor = get_preprocessor(config, split)
        data_aux_preprocessor = TestStandardPreProcessor(config)
        if split == 'val':
            dataset = _build_imglist_dataset(
                ood_config.name + '_' + split,
                split_config,
                ood_config,
                preprocessor,
                data_aux_preprocessor,
            )
            dataloader_dict[split] = DataLoader(
                dataset,
                batch_size=ood_config.batch_size,
                shuffle=ood_config.shuffle,
                num_workers=ood_config.num_workers,
            )
        else:
            sub_dataloader_dict = {}
            for dataset_name in split_config.datasets:
                dataset_config = split_config[dataset_name]
                dataset = _build_imglist_dataset(
                    ood_config.name + '_' + split,
                    dataset_config,
                    ood_config,
                    preprocessor,
                    data_aux_preprocessor,
                )
                sub_dataloader_dict[dataset_name] = DataLoader(
                    dataset,
                    batch_size=ood_config.batch_size,
                    shuffle=ood_config.shuffle,
                    num_workers=ood_config.num_workers,
                )
            dataloader_dict[split] = sub_dataloader_dict
    return dataloader_dict
