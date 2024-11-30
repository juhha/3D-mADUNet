# from .discriminator_loss
from .generator_loss import build_generator_loss
from .discriminator_loss import build_discriminator_loss

def build_loss(loss_type, opt):
    if loss_type == 'generator':
        return build_generator_loss(opt)
    elif loss_type == 'discriminator':
        return build_discriminator_loss(opt)