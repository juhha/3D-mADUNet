from .simpleunet import SimpleUNet, SimpleDiscriminator
from .attentiondenseunet import AttentionDenseUNet

def build_network(network_opt):
    net_type = network_opt['type']
    params = network_opt['params']
    if net_type == 'simpleunet':
        return SimpleUNet(**params)
    elif net_type == 'simpledisc':
        return SimpleDiscriminator(**params)
    elif net_type == 'attentiondenseunet':
        return AttentionDenseUNet(**params)