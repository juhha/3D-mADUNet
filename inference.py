from codes.data_utils import define_dataloaders
from codes.trainer import define_trainer

import os, yaml, json, argparse, nibabel as nib, tqdm
from monai.transforms import Invertd, SaveImage
from monai.data import decollate_batch
import torch
import matplotlib.pyplot as plt

# Load configurations
def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_file', type = str, default = 'test', dest = 'config_file',
                        help = 'configuration file name')
    parser.add_argument('--fold', type = int, default = 0, dest = 'fold',
                        help = 'i-th fold in k-fold cross validation')
    parser.add_argument('--num_workers', type = int, default = 8, dest = 'num_workers',
                        help = 'number of processing unit')
    parser.add_argument('--persistent', action='store_true', dest = 'persistent',
                        help = 'use persistent cache for data I/O')
    parser.add_argument('--memory_cache', action='store_true', dest = 'memory_cache',
                        help = 'use memory cache for data I/O')
    parser.add_argument('--device', type=str, default = 'cpu', dest = 'device',
                        help = 'gpu device')
    parser.add_argument('--progress', type=str, default = None, dest = 'progress',
                        help = 'report progress while training (pbar, print, wandb)')
    parser.add_argument('--pred_dir', type=str, default = None, dest = 'pred_dir',
                        help = 'prediction folder dictionary')
    return parser

def main():
    # don't know what this does, but this fixes problem of "RuntimeError: received 0 items of ancdata"
    torch.multiprocessing.set_sharing_strategy('file_system')
    # # Debug
    # base_dir = './'
    # parser = get_args()
    # args = parser.parse_known_args()[0]
    # args.config_file = 'attdenseunet_synth_task1_pelvis_nearest'
    # args.persistent = True
    # args.device = 'cuda:5'
    # args.progress = 'pbar'
    
    # path confiugrations
    base_dir = os.path.dirname(__file__)
    # get user arguments
    parser = get_args()
    args = parser.parse_args()
    
    data_dir = os.path.join(base_dir, 'data')
    config_dir = os.path.join(base_dir, 'options')
    
    # get configurations
    opts = yaml.safe_load(open(os.path.join(config_dir, f'{args.config_file}.yaml')))
    print(f'configurations: {opts}')
    opts['checkpoint'] = os.path.join(base_dir, 'checkpoint', args.config_file, f'fold_{args.fold}')
    opts = yaml.safe_load(open(os.path.join(opts['checkpoint'], 'config.yaml'), 'r'))
    # define dataloader
    dataloaders = define_dataloaders(data_dir, opts, args)
    # define trainer
    trainer = define_trainer(args, opts)

    # Evaluate
    dl = dataloaders['test']
    source_key = trainer.source_key
    target_key = trainer.target_key
    post_trans_pred = Invertd(keys = ['pred', target_key], transform = dl.dataset.transform, orig_keys = [target_key, target_key])
    device = trainer.device

    pbar = tqdm.tqdm(total = len(dl), position = 0, desc = 'evaluate')
    for batch in dl:
        source = batch[source_key].to(device)
        target = batch[target_key].to(device)
        with torch.no_grad():
            pred = trainer.predict(source, sliding_inference = True)
        pred.meta = target.meta
        batch['pred'] = pred
        list_data = decollate_batch(batch)
        for data in list_data:
            data = post_trans_pred(data)
        fileloc = os.path.dirname(os.path.join(trainer.cp_dir, 'test_output', data[f'fileloc_{target_key}']))
        if args.pred_dir is not None:
            fileloc = os.path.join(pred_dir, data[f'fileloc_{target_key}'])
        os.makedirs(fileloc, exist_ok = True)
        save_trans = SaveImage(output_dir = fileloc, output_postfix = '', output_ext = '.nii.gz', print_log = False, separate_folder = False)
        save_trans(data['pred'])
        pbar.update(1)
    pbar.close()

if __name__ == '__main__':
    main()