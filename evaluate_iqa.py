from codes.data_utils import define_dataloaders

import os, yaml, json, argparse, nibabel as nib, glob, tqdm, numpy as np
from monai.transforms import Invertd, CropForegroundd, Compose, LoadImaged, EnsureChannelFirstd
from monai.data import decollate_batch
import torch
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pred_dir', type = str, default = None, dest = 'pred_dir',
                        help = 'prediction file directory')
    parser.add_argument('--gt_dir', type = str, default = None, dest = 'gt_dir',
                        help = 'ground-truth file directory')
    parser.add_argument('--target_modality', type = str, default = 't2', dest = 'target_modality',
                        help = 'target modality - this is needed for intensity processing')
    parser.add_argument('--crop_foreground', action = 'store_true', dest = 'crop_foreground',
                        help = '(optional, but needed for SynthRAD2023) crop out background - this is needed only for foreground cropping (synthrad\'s CT dataset has artifacts)')
    parser.add_argument('--source_modality', type = str, default = 't1', dest = 'source_modality',
                        help = '(optional, but needed for SynthRAD2023) source modality - this is needed only for foreground cropping (synthrad\'s CT dataset has artifacts)')
    parser.add_argument('--result_path', type = str, default = None, dest = 'result_path',
                        help = 'result file directory')
    return parser

def normalized_mean_squared_error(gt, pred):
    mse = np.mean((gt - pred) ** 2)
    var = np.var(gt)
    return mse / var

def main():
    # path confiugrations
    base_dir = os.path.dirname(__file__)
    # get user arguments
    parser = get_args()
    args = parser.parse_args()
    result_path = args.result_path
    assert result_path, 'result_path argument is required'

    # # debug 1
    # base_dir = './'
    # parser = get_args()
    # args = parser.parse_known_args()[0]
    # args.pred_dir = './checkpoint/simpleunet_synth_task1_pelvis_nearest/fold_0/test_output'
    # args.gt_dir = './data/'
    # args.target_modality = 'ct'
    # args.source_modality = 'mr'
    # args.crop_foreground = True

    # # debug 2
    # base_dir = './'
    # parser = get_args()
    # args = parser.parse_known_args()[0]
    # args.pred_dir = './checkpoint/simpleunet_hcp1200_t1tot2_nearest/fold_0/test_output'
    # args.gt_dir = './data/'
    # args.target_modality = 'ct'
    # args.source_modality = 'mr'
    # args.crop_foreground = True

    pred_dir = args.pred_dir
    gt_dir = args.gt_dir
    target_modality = args.target_modality

    list_files_pred = glob.glob(os.path.join(args.pred_dir, '**', '*.nii.gz'), recursive = True)
    list_files_gt = [os.path.join(gt_dir, f.replace(pred_dir, '').strip('/')) for f in list_files_pred]
    trans = Compose([
        LoadImaged(keys = ['gt', 'pred'], allow_missing_keys = True),
        EnsureChannelFirstd(keys = ['gt', 'pred'])
    ])
    list_files_source = list_files_gt # dummy
    
    crop_foreground = args.crop_foreground
    if crop_foreground:
        # bring source modality
        source_modality = args.source_modality
        # crop foreground function
        trans = Compose([
            LoadImaged(keys = ['gt', 'pred', 'source']),
            EnsureChannelFirstd(keys = ['gt', 'pred', 'source']),
            CropForegroundd(keys = ['gt', 'pred', 'source'], source_key = 'gt'),
            CropForegroundd(keys = ['gt', 'pred', 'source'], source_key = 'source'),
        ])
        list_files_source = [os.path.join(os.path.dirname(f), os.path.basename(f).replace(target_modality, source_modality)) for f in list_files_gt]
    # inverse crop function
    invert_trans = Invertd(keys = ['gt', 'pred', 'source'], orig_keys=['gt', 'pred', 'source'], transform = trans, allow_missing_keys=True)
    
    performance = {}
    pbar = tqdm.tqdm(total = len(list_files_pred), position = 0)
    for file_pred, file_gt, file_source in zip(list_files_pred, list_files_gt, list_files_source):
        fdict = {'gt': file_gt, 'pred': file_pred, 'source': file_source}
        data = trans(fdict)
        # intensiry processing
        gt = data['gt'].as_tensor()
        pred = data['pred'].as_tensor()
        if target_modality == 'ct': # set min max value, then divide by max
            data['gt'].data = (gt.clip(-1024, 3000) + 1024) / (1024+3000) # for pred, this processing should already be applied.
        else: # make intensity between 0 to 1
            data['gt'].data = (gt - gt.min()) / (gt.max() - gt.min())
            data['pred'].data = (pred - pred.min()) / (pred.max() - pred.min())
        data = invert_trans(data)
        gt = data['gt'][0].numpy()
        pred = data['pred'][0].numpy()
        # calculate scores
        # 3d
        psnr = peak_signal_noise_ratio(gt, pred, data_range = 1)
        ssim = structural_similarity(gt, pred, data_range = 1)
        nmse = normalized_mean_squared_error(gt, pred)
        # # 2d - axial
        # psnr = np.mean([peak_signal_noise_ratio(g+1e-5,p+1e-5, data_range = 1) for g,p in zip(gt.transpose(2,0,1), pred.transpose(2,0,1))])
        # ssim = np.mean([structural_similarity(g+1e-5,p+1e-5, data_range = 1) for g,p in zip(gt.transpose(2,0,1), pred.transpose(2,0,1))])
        # nmse = np.mean([normalized_mean_squared_error(g+1e-5,p+1e-5) for g,p in zip(gt.transpose(2,0,1), pred.transpose(2,0,1))])

        # save
        fileloc = file_pred.replace(pred_dir, '').strip('/')
        performance[fileloc] = {
            'psnr': float(psnr),
            'ssim': float(ssim),
            'nmse': float(nmse)
        }
        
        pbar.update(1)
    pbar.close()
    os.makedirs(os.path.dirname(result_path), exist_ok = True)
    json.dump(performance, open(result_path, 'w'))

if __name__ == '__main__':
    main()