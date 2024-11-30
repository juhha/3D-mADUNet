import torch
import torch.nn as nn
import torch.nn.functional as F

def convert_to_dict(x):
    if isinstance(x, torch.Tensor):
        return {'level_0': x}
    return x

class BCEDiscLoss(nn.Module):
    def __init__(
        self,
        relativistic: bool = False,
        weight: dict = None
    ):
        super().__init__()
        self.relativistic = relativistic
        self.weight = weight
        self.loss_fn = nn.BCEWithLogitsLoss()
    def forward(self, fake, real = None, return_record = False):
        fake = convert_to_dict(fake)
        real = convert_to_dict(real)
        if self.weight is None:
            self.weight = {level: 1 for level in fake.keys()}
        elif isinstance(self.weight, (int, float)):
            self.weight = {level: self.weight for level in fake.keys()}
        loss = 0
        loss_record = {}
        for level in fake.keys():
            fake_ = fake[level].detach()
            real_ = real[level]
            if self.relativistic:
                l_real = self.loss_fn(real_ - torch.mean(fake_), torch.ones_like(fake_))
                l_fake = self.loss_fn(fake_ - torch.mean(real_), torch.zeros_like(fake_))
                l = (l_real + l_fake) / 2 * self.weight[level]
            else:
                l_real = self.loss_fn(real_, torch.ones_like(real_))
                l_fake = self.loss_fn(fake_, torch.zeros_like(fake_))
                l = (l_real + l_fake) * self.weight[level]
            loss += l
            loss_record[level] = {'fake': l_fake.item(), 'real': l_real.item()}
        if return_record:
            return loss, loss_record
        else:
            return loss

class WGanDiscLoss(nn.Module):
    def __init__(
        self,
        softplus: bool = False,
        weight = None,
    ):
        super().__init__()
        self.weight = weight
        self.softplus = softplus
    def forward(self, fake, real, return_record = False):
        fake = convert_to_dict(fake)
        real = convert_to_dict(real)
        if self.weight is None:
            self.weight = {level: 1 for level in out.keys()}
        elif isinstance(self.weight, (int, float)):
            self.weight = {level: self.weight for level in out.keys()}
        loss = 0
        loss_record = {}
        for level in out.keys():
            fake_ = fake[level]
            real_ = real[level]
            if self.softplus:
                l_real = -real_.mean() * self.weight[level]
                l_fake = fake_.mean() * self.weight[level]
            else:
                l_real = F.softplus(-real_).mean() * self.weight[level]
                l_fake = F.softplus(-fake_).mean() * self.weight[level]
            l = l_real + l_fake
            loss += l
            loss_record[level] = {'fake': l_fake.item(), 'real': l_real.item()}
        if return_record:
            return loss, loss_record
        else:
            return loss

class DiscriminatorLoss(nn.Module):
    def __init__(
        self,
        disc_loss_fn
    ):
        super().__init__()
        self.disc_loss_fn = disc_loss_fn
    def forward(self, out):
        fake = out['disc_fake']
        real = out['disc_real']
        loss = 0
        loss_record = {}
        # disc loss
        l, l_record = self.disc_loss_fn(fake, real, True)
        loss += l
        loss_record.update(l_record)
        return loss, loss_record

def build_discriminator_loss(opts):
    disc_loss = None
    disc_loss_opt = opts.get('disc_loss')
    if disc_loss_opt is not None:
        params = disc_loss_opt['params']
        weight_by_level = disc_loss_opt['weight_by_level']
        loss_type = params['loss_type']
        if loss_type == 'bce':
            relativistic = params['relativistic']
            disc_loss = BCEDiscLoss(relativistic = relativistic, weight = weight_by_level)
        elif loss_type == 'wgan':
            softplus = params.get('softplus')
            if softplus is None:
                softplus = False
            disc_loss = WGanDiscLoss(softplus = softplus, weight = weight_by_level)
    return DiscriminatorLoss(disc_loss_fn = disc_loss)
        