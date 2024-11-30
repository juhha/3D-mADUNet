import os, json, yaml, nibabel as nib, tqdm, time, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import Invertd, SaveImage

from .base_trainer import BaseTrainer
from codes.model import build_network
from codes.train_utils import build_optimizer, build_scaler, build_scheduler
# from codes.loss_utils import build_loss
from codes.loss import build_loss

def collect_keys(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(collect_keys(v, new_key, sep=sep))
        elif isinstance(v, float):
            items.append((new_key, v))
    return items

class SRGANTrainer(BaseTrainer):
    def __init__(
        self,
        opts,
        args
    ):
        super().__init__(opts, args)
    def initialize_train(self):
        if not os.path.exists(os.path.join(self.cp_dir, 'config.yaml')):
            os.makedirs(self.cp_dir, exist_ok = True)
            yaml.safe_dump(self.opts, open(os.path.join(self.cp_dir, 'config.yaml'), 'w'))
        ##########
        # define networks
        net_opt = self.opts['network_opt']
        self.net_g = build_network(net_opt['net_g']).to(self.device)
        self.net_d = build_network(net_opt['net_d']).to(self.device)
        
        ##########
        # define optimizer
        optim_opt = self.opts['optim_opt']
        self.optimizer_g = build_optimizer(self.net_g.parameters(), optim_opt['generator'])
        self.optimizer_d = build_optimizer(self.net_d.parameters(), optim_opt['discriminator'])
        
        # ##########
        # # define scaler
        scaler_opt = self.opts['scaler_opt']
        self.scaler_g = build_scaler(scaler_opt['use_scaler'], scaler_opt['generator'])
        self.scaler_d = build_scaler(scaler_opt['use_scaler'], scaler_opt['discriminator'])

        ##########
        # define scheduler
        scheduler_opt = self.opts['scheduler_opt']
        self.scheduler_g = build_scheduler(self.optimizer_g, scheduler_opt['generator'])
        self.scheduler_d = build_scheduler(self.optimizer_d, scheduler_opt['discriminator'])

        ##########
        # define mixed precision
        precision = self.opts['autocast'].get('precision')
        self.precision = torch.float16 if precision == 'float16' else torch.float32
        self.autocast_enabled = self.opts['autocast']['enabled']
        
        ##########
        # define loss functions
        loss_opt = self.opts['loss_opt']
        # generator loss functions
        # if 'perception_loss' in loss_opt['generator'].keys():
        #     loss_opt['generator']['perception_loss']['params']['device'] = self.device
        self.loss_fn_g = build_loss('generator', loss_opt['generator']).to(self.device)
        # discriminator loss functions
        self.loss_fn_d = build_loss('discriminator', loss_opt['discriminator'])
        
        ##########
        # load from checkpoint
        if os.path.exists(os.path.join(self.cp_dir, 'latest.pt')):
            state_dict = torch.load(os.path.join(self.cp_dir, 'latest.pt'), map_location = self.device)
            self.net_g.load_state_dict(state_dict['net_g'])
            self.net_d.load_state_dict(state_dict['net_d'])
            self.optimizer_g.load_state_dict(state_dict['optimizer_g'])
            self.optimizer_d.load_state_dict(state_dict['optimizer_d'])
            self.scheduler_g.load_state_dict(state_dict['scheduler_g'])
            self.scheduler_d.load_state_dict(state_dict['scheduler_d'])
            self.scaler_g.load_state_dict(state_dict['scaler_g'])
            self.scaler_d.load_state_dict(state_dict['scaler_d'])
            self.curr_epoch = state_dict['curr_epoch']
            self.best_epoch = state_dict['best_epoch']
            self.best_score = state_dict['best_score']
            if os.path.exists(os.path.join(self.cp_dir, 'progress.json')):
                self.progress = json.load(open(os.path.join(self.cp_dir, 'progress.json'), 'r'))
            else:
                self.progress = {
                    'train': {
                        'loss': {}
                    },
                    # 'val': {
                    #     'loss': {}
                    # }
                }
        else:
            self.curr_epoch = 0
            self.best_epoch = 0
            self.best_score = 0
            self.progress = {
                'train': {
                    'loss': {}
                },
                # 'val': {
                #     'loss': {}
                # }
            }
    def convert_to_dict(self, x, clip = False):
        if isinstance(x, torch.Tensor):
            x = {'level_0': x.clip(0,1)}
        if clip:
            for key in x.keys():
                x[key] = x[key].clip(0,1)
        return x
    @torch.no_grad()
    def predict(self, x, sliding_inference = False):
        x = x.to(self.device)
        # basic forward
        if sliding_inference:
            patch_size = self.opts['train_opt']['patch_size']
            sw_batch_size = self.opts['train_opt']['batch_size'] * self.opts['train_opt']['num_patch']
            out_g = sliding_window_inference(x, roi_size = patch_size, sw_batch_size = sw_batch_size, predictor = self.net_g, overlap = 0.625, mode = 'gaussian', sigma_scale = 0.125, sw_device = self.device, device = self.device)
            out_g = self.convert_to_dict(out_g, clip = True)['level_0'].cpu().detach()
        else:
            out_g = self.net_g(x)
            out_g = self.convert_to_dict(out_g, clip = True)['level_0'].cpu().detach()
        return out_g
    @torch.no_grad()
    def save_sample_image(self, source, target, pred, save_fpath):
        s = source[0,0].cpu().detach()
        t = target[0,0].cpu().detach()
        p = pred[0,0].cpu().detach()
        fig, axes = plt.subplots(3,3,figsize = (9,9))
        for i in range(3):
            s = s.permute(1,2,0)
            t = t.permute(1,2,0)
            p = p.permute(1,2,0)
            axes[i,0].imshow(s[s.shape[0]//2], cmap = 'gray'); axes[i,0].axis('off')
            axes[i,1].imshow(t[t.shape[0]//2], cmap = 'gray'); axes[i,1].axis('off')
            axes[i,2].imshow(p[p.shape[0]//2], cmap = 'gray'); axes[i,2].axis('off')
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_fpath), exist_ok = True)
        fig.savefig(save_fpath)
        plt.close()
        del fig
    def train_iter(self, batch):
        # load data
        source = batch[self.source_key].to(self.device)
        target = batch[self.target_key].to(self.device)
        target_detached = target.detach() if isinstance(target, torch.Tensor) else {key:val.detach() for key, val in target.items()}
        # basic forward
        out_g = self.net_g(source)
        out_g = self.convert_to_dict(out_g, clip = True)
        out_g_detached = {key: out_g[key].detach() for key in out_g.keys()}
        # update discriminator
        for p in self.net_d.parameters():
            p.requires_grad = True
        # forward
        out_d = {'disc_fake': self.net_d(out_g_detached), 'disc_real': self.net_d(target)}
        # backward
        with torch.cuda.amp.autocast(enabled = self.autocast_enabled, dtype = self.precision):
            self.optimizer_d.zero_grad()
            loss_d, loss_d_record = self.loss_fn_d(out_d)
            self.scaler_d.scale(loss_d).backward()
            self.scaler_d.step(self.optimizer_d)
            self.scaler_d.update()
        # update generator
        for p in self.net_d.parameters():
            p.requires_grad = False
        # forward
        out_g = {'disc_real': self.net_d(out_g), 'disc_fake': self.net_d(target_detached), 'out': out_g}
        # backward
        with torch.cuda.amp.autocast(enabled = self.autocast_enabled, dtype = self.precision):
            self.optimizer_g.zero_grad()
            loss_g, loss_g_record = self.loss_fn_g(out_g, target)
            self.scaler_g.scale(loss_g).backward()
            self.scaler_g.step(self.optimizer_g)
            self.scaler_g.update()
        # record loss
        loss_record = {
            'g': loss_g_record,
            'd': loss_d_record
        }
        loss_record = collect_keys(loss_record)
        for key, val in loss_record:
            self.train_loss.setdefault(key, [])
            self.train_loss[key].append(val)
    def train_epoch(self, dl):
        start_time = time.time()
        self.train_loss = {} # initialize train progress loss
        max_epoch = self.opts['train_opt']['max_epoch']
        self.net_g.train()
        if self.args.progress == 'pbar':
            pbar = tqdm.tqdm(total = len(dl), position = 0, desc = f'train ({self.curr_epoch}/{max_epoch})')
        for batch in dl:
            self.train_iter(batch)
            if self.args.progress == 'pbar':
                pbar.update(1)
        self.curr_epoch += 1
        end_time = time.time()
        if self.args.progress == 'pbar':
            pbar.close()
        # save state_dict
        state_dict = {
            'net_g': self.net_g.state_dict(),
            'net_d': self.net_d.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
            'scheduler_g': self.scheduler_g.state_dict(),
            'scheduler_d': self.scheduler_d.state_dict(),
            'scaler_g': self.scaler_g.state_dict(),
            'scaler_d': self.scaler_d.state_dict(),
            'curr_epoch': self.curr_epoch,
            'best_epoch': self.best_epoch,
            'best_score': self.best_score
        }
        torch.save(state_dict, os.path.join(self.cp_dir, 'latest.pt'))
        # save progress
        for key,val in self.train_loss.items():
            self.progress['train']['loss'].setdefault(key, [])
            self.progress['train']['loss'][key].append(float(np.mean(val)))
        json.dump(self.progress, open(os.path.join(self.cp_dir, 'progress.json'), 'w'))
        if self.args.progress in ('print', 'pbar'):
            print(f'Train ({self.curr_epoch}/{max_epoch}) took {end_time - start_time:,.2f} seconds.')
            print('Train Loss')
            for key,val in self.progress['train']['loss'].items():
                print(f'\t{key}:{val[-1]:.3f}')
        # save output image for last batch
        image_path = os.path.join(self.cp_dir, 'train_samples', f'{str(self.curr_epoch).zfill(5)}.png')
        source = batch[self.source_key]
        target = batch[self.target_key]
        pred = self.predict(source, sliding_inference = False)
        self.save_sample_image(source, target, pred, image_path)
    @torch.no_grad()
    def eval_iter(self, batch):
        # save output slide image
        image_path = os.path.join(self.cp_dir, 'eval_samples', os.path.splitext(batch[f'fileloc_{self.target_key}'][0])[0], f'{str(self.curr_epoch).zfill(5)}.png')
        os.makedirs(os.path.dirname(image_path), exist_ok = True)
        source = batch[self.source_key].to(self.device)
        target = batch[self.target_key].to(self.device)
        pred = self.predict(source, sliding_inference = True)
        # # calculate loss
        # loss_g, loss_g_record = self.loss_fn_g(pred, target)
        # loss_record = {'g': loss_g_record}
        # # record loss
        # loss_record = collect_keys(loss_record)
        # for key, val in loss_record:
        #     self.eval_loss.setdefault(key, [])
        #     self.eval_loss[key].append(val)
        pred.meta = target.meta
        self.save_sample_image(source, target, pred, image_path)
        # save output nifti file
        batch['pred'] = pred
        list_data = decollate_batch(batch)
        for data in list_data:
            if self.post_trans is not None:
                data = self.post_trans(data)
            fileloc = os.path.dirname(os.path.join(self.cp_dir, 'eval_nifti', f'{str(self.curr_epoch).zfill(5)}', data[f'fileloc_{self.target_key}']))
            save_trans = SaveImage(output_dir = fileloc, output_postfix = '', output_ext = '.nii.gz', print_log = False, separate_folder = False)
            save_trans(data['pred'])
    @torch.no_grad()
    def eval(self, dl):
        start_time = time.time()
        self.net_g.eval()
        self.post_trans = None
        # self.eval_loss = {}
        try:
            self.post_trans = Invertd(keys = 'pred', transform = dl.dataset.transform, orig_keys = self.target_key)
        except:
            pass
        max_epoch = self.opts['train_opt']['max_epoch']
        if self.args.progress == 'pbar':
            pbar = tqdm.tqdm(total = len(dl), position = 0, desc = f'val ({self.curr_epoch}/{max_epoch})')
        for batch in dl:
            self.eval_iter(batch)
            if self.args.progress == 'pbar':
                pbar.update(1)
        if self.args.progress == 'pbar':
            pbar.close()
        end_time = time.time()
        # save progress
        # for key,val in self.eval_loss.items():
        #     self.progress['val']['loss'].setdefault(key, [])
        #     self.progress['val']['loss'][key].append(float(np.mean(val)))
        # json.dump(self.progress, open(os.path.join(self.cp_dir, 'progress.json'), 'w'))
        if self.args.progress in ('print', 'pbar'):
            print(f'Eval ({self.curr_epoch}/{max_epoch}) took {end_time - start_time:,.2f} seconds.')
            # print('Val Loss')
            # for key,val in self.progress['val']['loss'].items():
            #     print(f'\t{key}:{val[-1]:.3f}')
    @torch.no_grad()
    def test_iter(self, batch):
        source = batch[self.source_key]
        target = batch[self.target_key]
        pred = self.predict(source, sliding_inference = True)
        pred.meta = target.meta
        # save output nifti file
        batch['pred'] = pred
        list_data = decollate_batch(batch)
        for data in list_data:
            if self.post_trans is not None:
                data = self.post_trans(data)
            fileloc = os.path.dirname(os.path.join(self.cp_dir, 'test_output', data[f'fileloc_{self.target_key}']))
            save_trans = SaveImage(output_dir = fileloc, output_postfix = '', output_ext = '.nii.gz', print_log = False, separate_folder = False)
            save_trans(data['pred'])
    @torch.no_grad()
    def test(self, dl):
        self.net_g.eval()
        self.post_trans = None
        if not hasattr(self, 'post_trans'):
            try:
                self.post_trans = Invertd(keys = 'pred', transforms = dl.dataset.transforms, orig_keys = self.target_key)
            except:
                pass
        max_epoch = self.opts['train_opt']['max_epoch']
        if self.args.progress == 'pbar':
            pbar = tqdm.tqdm(total = len(dl), position = 0, desc = f'test ({self.curr_epoch}/{max_epoch})')
        for batch in dl:
            self.test_iter(batch)
            if self.args.progress == 'pbar':
                pbar.update(1)
        if self.args.progress == 'pbar':
            pbar.close()
    def save_checkpoint(self, save_fname = 'latest.pt'):
        pass

if __name__ == '__main__':
    import argparse
    args = argparse.Namespace()
    args.device = 'cuda'
    opts = {}
    base_dir = '../../'
    SRGANTrainer(base_dir, opts, args)