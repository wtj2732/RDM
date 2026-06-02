import torch
import torch.backends.cudnn as cudnn
import openood.utils.comm as comm
from .resnet18_32x32 import ResNet18_32x32


def _load_state_dict(net, checkpoint):
    loaded = torch.load(checkpoint, map_location='cpu')
    if isinstance(loaded, dict) and 'state_dict' in loaded:
        loaded = loaded['state_dict']
    try:
        net.load_state_dict(loaded, strict=False)
    except RuntimeError:
        # Common case: checkpoint fc layer shape differs. Load everything else.
        if isinstance(loaded, dict):
            loaded = dict(loaded)
            loaded.pop('fc.weight', None)
            loaded.pop('fc.bias', None)
        net.load_state_dict(loaded, strict=False)


def get_network(network_config):
    num_classes = network_config.num_classes
    if network_config.name == 'resnet18_32x32':
        net = ResNet18_32x32(num_classes=num_classes)
    else:
        raise Exception(f'Unsupported network in minimal build: {network_config.name}')

    if network_config.pretrained:
        _load_state_dict(net, network_config.checkpoint)
        print(f'Model Loading {network_config.name} Completed!')

    if network_config.num_gpus > 1:
        net = torch.nn.parallel.DistributedDataParallel(
            net.cuda(),
            device_ids=[comm.get_local_rank()],
            broadcast_buffers=True,
        )
    elif network_config.num_gpus > 0:
        net.cuda()

    cudnn.benchmark = True
    return net
