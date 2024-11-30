import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor

def convert_to_dict(x):
    if isinstance(x, torch.Tensor):
        return {'level_0': x}
    return x

class PixelLoss(nn.Module):
    def __init__(
        self,
        loss_type: str,
        weight: dict = None
    ):
        super().__init__()
        self.loss_type = loss_type
        self.weight = weight
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss()
    def forward(self, out, target, return_record = False):
        out = convert_to_dict(out)
        target = convert_to_dict(target)
        if self.weight is None:
            self.weight = {level: 1 for level in out.keys()}
        elif isinstance(self.weight, (int, float)):
            self.weight = {level: self.weight for level in out.keys()}
        loss = 0
        loss_record = {}
        for level in out.keys():
            o = out[level]
            if level in target:
                t = target[level]
            else:
                t = F.interpolate(target['level_0'], size = o.shape[2:], mode = 'nearest')
            l = self.loss_fn(o, t) * self.weight[level]
            loss += l
            loss_record[level] = l.item()
        if return_record:
            return loss, loss_record
        else:
            return loss

class BCEAdvLoss(nn.Module):
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
        if self.relativistic:
            real = convert_to_dict(real)
        if self.weight is None:
            self.weight = {level: 1 for level in fake.keys()}
        elif isinstance(self.weight, (int, float)):
            self.weight = {level: self.weight for level in fake.keys()}
        loss = 0
        loss_record = {}
        for level in fake.keys():
            fake_ = fake[level]
            if self.relativistic:
                real_ = real[level].detach()
                l_real = self.loss_fn(real_ - torch.mean(fake_), torch.zeros_like(fake_))
                l_fake = self.loss_fn(fake_ - torch.mean(real_), torch.ones_like(fake_))
                l = (l_real + l_fake) / 2 * self.weight[level]
            else:
                l = self.loss_fn(fake_, torch.ones_like(fake_)) * self.weight[level]
            loss += l
            loss_record[level] = l.item()
        if return_record:
            return loss, loss_record
        else:
            return loss

class WGanAdvLoss(nn.Module):
    def __init__(
        self,
        softplus: bool = False,
        weight = None,
    ):
        super().__init__()
        self.weight = weight
        self.softplus = softplus
    def forward(self, fake, real = None, return_record = False):
        fake = convert_to_dict(fake)
        if self.weight is None:
            self.weight = {level: 1 for level in out.keys()}
        elif isinstance(self.weight, (int, float)):
            self.weight = {level: self.weight for level in out.keys()}
        loss = 0
        loss_record = {}
        for level in out.keys():
            fake_ = fake[level]
            if self.softplus:
                l = -fake_.mean() * self.weight[level]
            else:
                l = F.softplus(-fake_).mean() * self.weight[level]
            loss += l
            loss_record[level] = l.item()
        if return_record:
            return loss, loss_record
        else:
            return loss

class AdvLoss(nn.Module):
    def __init__(
        self,
        loss_fn
    ):
        super().__init__()
        self.loss_fn = loss_fn
    def forward(self, out):
        fake = out['disc_fake']
        real = out['disc_real']
        loss = 0
        loss_record = {}
        # disc loss
        l, l_record = self.loss_fn(fake, real, True)
        loss += l
        loss_record.update(l_record)
        return loss, loss_record

class PerceptionLoss3D(nn.Module):
    def __init__(
        self,
        feature_extractor: "nn.Module",
        loss_fn = nn.L1Loss(),
        channel_dim: int = 3,
        normalize_mean: list = [0.485, 0.456, 0.406],
        normalize_std: list = [0.229, 0.224, 0.225],
        separate_channel: bool = True,
        base_weight: float = 1.0,
        feature_weights: dict = None,
        views: str = ['axial', 'sagittal', 'coronal']
    ):
        super().__init__()
        self.views = views
        self.loss_fn = loss_fn
        self.feature_extractor = feature_extractor
        self.feature_extractor.eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.channel_dim = channel_dim
        if normalize_mean is not None:
            c = len(normalize_mean)
            self.normalize_mean = torch.Tensor(normalize_mean).view(1,c,1,1,1)
            self.normalize_std = torch.Tensor(normalize_std).view(1,c,1,1,1)
        else:
            self.normalize_mean = normalize_mean
            self.normalize_std = normalize_std
        self.separate_channel = separate_channel
        self.base_weight = base_weight
    def forward(self, out, target, return_record = False):
        if isinstance(out, dict):
            out = out['level_0']
        if isinstance(target, dict):
            target = target['level_0']
        if self.normalize_mean is not None:
            device = out.device
            self.normalize_mean = self.normalize_mean.to(device)
            self.normalize_std = self.normalize_std.to(device)
            out = (out - self.normalize_mean) / (self.normalize_std + 1e-5)
            target = (target - self.normalize_mean) / (self.normalize_std + 1e-5)
        # init loss variable
        loss = 0
        # seperate channel dim into batch dim if needed
        b,c,h,w,z = out.shape
        if c != self.channel_dim or self.separate_channel:
            b_orig,c_orig,h,w,z = out.shape
            out = out.view(b_orig*c_orig,1,h,w,z)
            target = target.view(b_orig*c_orig,1,h,w,z)
            out = out.repeat(1,self.channel_dim,1,1,1)
            target = target.repeat(1,self.channel_dim,1,1,1)
        # Axial view
        if 'axial' in self.views:
            # make 3D->2D (axial view)
            b,c,h,w,z = out.shape
            o = out.permute(0,4,1,2,3).reshape(b*z,c,h,w)
            t = target.permute(0,4,1,2,3).reshape(b*z,c,h,w)
            # make channel dimension feasible to perception model
            o_features = self.feature_extractor(o)
            t_features = self.feature_extractor(t)
            for key in o_features.keys():
                b_z,c,h,w = o_features[key].shape
                z = b_z // b
                o_feat = o_features[key].reshape(b,z,c,h,w).permute(0,2,3,4,1) # make b,c,h,w,z again
                t_feat = t_features[key].reshape(b,z,c,h,w).permute(0,2,3,4,1) # make b,c,h,w,z again
                loss += self.loss_fn(o_feat, t_feat) / self.base_weight
        # sagittal view
        if 'sagittal' in self.views:
            # make 3D->2D (sagittal view)
            b,c,h,w,z = out.shape
            o = out.permute(0,3,1,2,4).reshape(b*w,c,h,z)
            t = target.permute(0,3,1,2,4).reshape(b*w,c,h,z)
            # make channel dimension feasible to perception model
            o_features = self.feature_extractor(o)
            t_features = self.feature_extractor(t)
            for key in o_features.keys():
                b_w,c,h,z = o_features[key].shape
                w = b_w // b
                o_feat = o_features[key].reshape(b,w,c,h,z).permute(0,2,3,1,4) # make b,c,h,w,z again
                t_feat = t_features[key].reshape(b,w,c,h,z).permute(0,2,3,1,4) # make b,c,h,w,z again
                loss += self.loss_fn(o_feat, t_feat) / self.base_weight
        if 'coronal' in self.views:
            # Coronal view
            # make 3D->2D (coronal view)
            b,c,h,w,z = out.shape
            o = out.permute(0,2,1,3,4).reshape(b*h,c,w,z)
            t = target.permute(0,2,1,3,4).reshape(b*h,c,w,z)
            # make channel dimension feasible to perception model
            o_features = self.feature_extractor(o)
            t_features = self.feature_extractor(t)
            for key in o_features.keys():
                b_h,c,w,z = o_features[key].shape
                h = b_h // b
                o_feat = o_features[key].reshape(b,h,c,w,z).permute(0,2,1,3,4) # make b,c,h,w,z again
                t_feat = t_features[key].reshape(b,h,c,w,z).permute(0,2,1,3,4) # make b,c,h,w,z again
                loss += self.loss_fn(o_feat, t_feat) / self.base_weight
            loss = loss / len(self.views)
            loss_record = loss.item()
            if return_record:
                return loss, loss_record
            else:
                return loss

def define_pretrained(model_name):
    if model_name == 'vgg19':
        pretrained = torchvision.models.vgg19(weights = torchvision.models.vgg.VGG19_Weights.IMAGENET1K_V1)
    elif model_name == 'resnet152':
        pretrained = torchvision.models.resnet152(weights = torchvision.models.resnet.ResNet152_Weights)
    elif model_name == 'inception_v3':
        pretrained = torchvision.models.inception_v3(weights = torchvision.models.inception.Inception_V3_Weights)
    return pretrained

class GeneratorLoss(nn.Module):
    def __init__(
        self,
        pixel_loss_fn = None,
        perception_loss_fn = None,
        adv_loss_fn = None,
        weight: dict = None
    ):
        super().__init__()
        self.pixel_loss_fn = pixel_loss_fn
        self.perception_loss_fn = perception_loss_fn
        self.adv_loss_fn = adv_loss_fn
        self.weight = weight if weight is not None else {}
    def forward(self, out, target):
        loss = 0
        loss_record = {}
        # pixel loss
        if self.pixel_loss_fn is not None:
            o = out['out']
            w = self.weight['pixel'] if 'pixel' in self.weight else 1
            l, l_record = self.pixel_loss_fn(o, target, True)
            loss += l * w
            loss_record['pixel'] = l_record
        # perception loss
        if self.perception_loss_fn is not None:
            o = out['out']
            w = self.weight['perception'] if 'perception' in self.weight else 1
            l, l_record = self.perception_loss_fn(o, target, True)
            loss += l * w
            loss_record['perception'] = l_record
        # adv loss
        if self.adv_loss_fn is not None:
            w = self.weight['adv'] if 'adv' in self.weight else 1
            l, l_record = self.adv_loss_fn(out)
            loss += l * w
            loss_record['adv'] = l_record
        return loss, loss_record

def build_generator_loss(opts):
    pixel_loss = None
    perception_loss = None
    adv_loss = None
    tv_loss = None
    weight_by_loss = {}
    # get settings
    pixel_loss_opt = opts.get('pixel_loss')
    perception_loss_opt = opts.get('perception_loss')
    adv_loss_opt = opts.get('adv_loss')
    # pixel loss
    if pixel_loss_opt is not None:
        weight_by_level = pixel_loss_opt.get('weight_by_level')
        params = pixel_loss_opt['params']
        pixel_loss = PixelLoss(weight = weight_by_level, **params)
        weight_by_loss['pixel'] = pixel_loss_opt['weight']
    # perception loss
    if perception_loss_opt is not None: # only applied to the final output level
        loss_type = perception_loss_opt['type']
        params = perception_loss_opt['params']
        model_name = params['model_name']
        return_nodes = params['return_nodes']
        channel_dim = 3
        separate_channel = True
        base_weight = len(return_nodes)
        pretrained = define_pretrained(model_name).eval()
        feature_extractor = create_feature_extractor(pretrained, return_nodes)
        if loss_type == '3d':
            perception_loss = PerceptionLoss3D(feature_extractor, nn.L1Loss(), channel_dim = channel_dim, separate_channel = separate_channel, base_weight = base_weight)
        weight_by_loss['perception'] = perception_loss_opt['weight']
    # adv loss
    if adv_loss_opt is not None:
        weight_by_level = adv_loss_opt.get('weight_by_level')
        params = adv_loss_opt['params']
        loss_type = params['loss_type']
        if loss_type == 'bce':
            relativistic = params['relativistic']
            loss_fn = BCEAdvLoss(relativistic = relativistic, weight = weight_by_level)
        elif loss_type == 'wgan':
            softplus = params.get('softplus')
            if softplus is None:
                softplus = False
            loss_fn = WGanAdvLoss(softplus = softplus, weight = weight_by_level)
        adv_loss = AdvLoss(loss_fn)
        weight_by_loss['adv'] = adv_loss_opt['weight']
    return GeneratorLoss(pixel_loss_fn = pixel_loss, perception_loss_fn = perception_loss, adv_loss_fn = adv_loss, weight = weight_by_loss)