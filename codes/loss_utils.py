import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.models.feature_extraction import create_feature_extractor

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
    def forward(self, out, target):
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
            return loss / len(self.views)

def define_pretrained(model_name):
    if model_name == 'vgg19':
        pretrained = torchvision.models.vgg19(weights = torchvision.models.vgg.VGG19_Weights.IMAGENET1K_V1)
    elif model_name == 'resnet152':
        pretrained = torchvision.models.resnet152(weights = torchvision.models.resnet.ResNet152_Weights)
    elif model_name == 'inception_v3':
        pretrained = torchvision.models.inception_v3(weights = torchvision.models.inception.Inception_V3_Weights)
    return pretrained

class BCEDiscLoss(nn.Module):
    def __init__(
        self
    ):
        super().__init__()
    def forward(self, x_real, x_fake):
        target_real = torch.ones_like(x_real)
        target_fake = torch.zeros_like(x_fake)
        return {
            'real': F.binary_cross_entropy_with_logits(x_real, target_real),
            'fake': F.binary_cross_entropy_with_logits(x_fake, target_fake),
        }

class BCEAdvLoss(nn.Module):
    def __init__(
        self
    ):
        super().__init__()
    def forward(self, x_fake):
        target_fake = torch.ones_like(x_fake)
        return {
            'adv': F.binary_cross_entropy_with_logits(x_fake, target_fake),
        }

class DiscriminatorLoss(nn.Module):
    def __init__(
        self,
        loss_opt
    ):
        super().__init__()
        loss_type = loss_opt['disc_loss']['type']
        loss_params = loss_opt['disc_loss']['params']
        weights = loss_opt['disc_loss']['weight']
        if loss_type == 'bce':
            self.loss_fn = BCEDiscLoss(**loss_params)
    def forward(self, x: dict):
        loss_dict = {}
        loss = 0
        for key in weights.keys():
            w = weights[key]
            if w > 0:
                l = self.loss_fn(x['real'][key], x['fake'][key])
                if isinstance(l, dict):
                    loss_dict[key] = {}
                    for k, val in l.items():
                        val = val * w
                        loss += val
                        loss_dict[key][k] = val.item()
                else:
                    l = l * w
                    loss += l
                    loss_dict[key] = l.item()
        return loss, loss_dict

class HaarWaveletTransform3D(nn.Module):
    def __init__(
        self,
        scale: int = 1
    ):
        super().__init__()
        self.scale = scale
        # Make weights for wavelet kernels
        H0 = torch.Tensor([1, 1])
        H1 = torch.Tensor([1, -1])
        h1d = [H0,H1]
        # Create Haar wavelet filterbanks
        h2d = []
        for h1 in h1d:
            for h2 in h1d:
                h2d.append(torch.outer(h1,h2))

        h3d = []
        for h1_2d in h2d:
            for h2_1d in h1d:
                h3d.append(torch.outer(h1_2d.flatten(), h2_1d).reshape(2,2,2))
        h3d = torch.stack(h3d).unsqueeze(1)/8
        if scale == 1:
            h3d = F.pad(h3d, pad = [1,0,1,0,1,0])
        self.kernel = h3d
    def forward(self, x):
        if self.kernel.device != x.device:
            self.kernel = self.kernel.to(x.device)
        if self.scale == 1:
            return F.conv3d(x, self.kernel, padding = 1)
        else:
            return F.conv3d(x, self.kernel, stride = self.scale, padding = 0)

class WaveletLoss(nn.Module):
    def __init__(
        self,
        scale: int = 1,
        weights = [0.02, 0.08, 0.07, 0.2 , 0.06, 0.18, 0.11, 0.29]
    ):
        super().__init__()
        self.weights = torch.Tensor(weights).view(1,8,1,1,1)
        self.trans = HaarWaveletTransform3D(scale = scale)
    def forward(self, out, target):
        out_wav = self.trans(out)
        target_wav = self.trans(target)
        if self.weights.device != out.device:
            self.weights = self.weights.to(out.device)
        loss = (F.l1_loss(out_wav, target_wav, reduction = 'none') * self.weights).mean()
        return loss

class GeneratorLoss(nn.Module):
    def __init__(
        self,
        loss_opt
    ):
        super().__init__()
        self.apply_pixel_loss = False
        self.apply_perception_loss = False
        self.apply_adv_loss = False
        # define pixel loss
        if 'pixel_loss' in loss_opt.keys():
            loss_type = loss_opt['pixel_loss']['type']
            loss_params = loss_opt['pixel_loss']['params']
            self.weights_pixel = loss_opt['pixel_loss']['weight']
            if loss_type == 'l1':
                self.loss_fn_pixel = nn.L1Loss()
            elif loss_type == 'wavelet':
                self.loss_fn_pixel = WaveletLoss(**loss_params)
            self.apply_pixel_loss = True
        # define percetion loss - only for level_0
        if 'perception_loss' in loss_opt.keys():
            loss_type = loss_opt['perception_loss']['type']
            loss_params = loss_opt['perception_loss']['params']
            self.weights_perception = loss_opt['perception_loss']['weight']
            return_nodes = loss_params['return_nodes'] if 'return_nodes' in loss_params.keys() else ['features.35']
            model_name = loss_params['model_name']
            device = loss_params['device']
            channel_dim = 3
            separate_channel = True
            base_weight = len(return_nodes)
            pretrained = define_pretrained(model_name).eval()
            feature_extractor = create_feature_extractor(pretrained, return_nodes)
            self.loss_fn_perception = PerceptionLoss3D(feature_extractor, nn.L1Loss(), channel_dim = channel_dim, separate_channel = separate_channel, base_weight = base_weight).to(device)
            self.apply_perception_loss = True
        # define adversarial loss
        if 'advloss' in loss_opt.keys():
            loss_type = loss_opt['adv_loss']['type']
            loss_params = loss_opt['adv_loss']['params']
            self.weights_adv = loss_opt['adv_loss']['weight']
            if loss_type == 'bce':
                self.loss_fn_adv = BCEAdvLoss(**loss_params)
            self.apply_adv_loss = True
    def forward(self, out:dict, target):
        loss_dict = {}
        loss = 0
        # pixel loss
        if self.apply_pixel_loss:
            loss_dict['pixel'] = {}
            for key in self.weight_pixel.keys():
                w = self.weight_pixel[key]
                if w > 0:
                    o = out[key]
                    t = F.interpolate(target, size = o.shape[-3:], mode = 'nearest')
                    l = self.loss_fn_pixel(o, t) * w
                    loss += l
                    loss_dict['pixel'][key] = l.item()
        # perception loss
        if self.apply_perception_loss:
            loss_dict['perception'] = {}
            w = self.weight_perception
            l = self.loss_fn_perception(out['level_0'], target) * w
            loss += l
            loss_dict['perception']['level_0'] = l.item()
        # adv loss
        if self.apply_adv_loss:
            loss_dict['adv'] = {}
            for key in self.weight_adv.keys():
                w = self.weight_adv[key]
                if w > 0:
                    o = out[key]
                    l = self.loss_fn_adv(o) * 2
                    loss += l
                    loss_dict['adv'][key] = l.item()
        return loss, loss_dict

def build_loss(loss_type, loss_opt):
    if loss_type == 'generator':
        return GeneratorLoss(loss_opt)
    elif loss_type == 'discriminator':
        return DiscriminatorLoss(loss_opt)