# from codes.model_utils import define_model
from codes.data_utils import define_dataloaders
# from codes.train_utils import define_loss, define_optimizer, define_scheduler, define_scaler
from codes.trainer import define_trainer

import os, yaml, json, argparse
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
    return parser

def main():
    # don't know what this does, but this fixes problem of "RuntimeError: received 0 items of ancdata"
    torch.multiprocessing.set_sharing_strategy('file_system')
    # path confiugrations
    base_dir = os.path.dirname(__file__)
    # get user arguments
    parser = get_args()
    args = parser.parse_args()
    
    # # debug
    # base_dir = './'
    # parser = get_args()
    # args = parser.parse_args(['--config_file', 'attdenseunet_synth_task1_pelvis_nearest', '--persistent', '--device', 'cuda:7', '--progress', 'pbar'])
    
    data_dir = os.path.join(base_dir, 'data')
    config_dir = os.path.join(base_dir, 'options')
    
    # get configurations
    opts = yaml.safe_load(open(os.path.join(config_dir, f'{args.config_file}.yaml')))
    print(f'configurations: {opts}')
    opts['checkpoint'] = os.path.join(base_dir, 'checkpoint', args.config_file, f'fold_{args.fold}')
    # define dataloader
    dataloaders = define_dataloaders(data_dir, opts, args)
    # define trainer
    trainer = define_trainer(args, opts)
    ###########
    # train
    for epoch in range(trainer.curr_epoch, trainer.max_epoch):
        # train one epoch
        trainer.train_epoch(dataloaders['train'])
        if trainer.curr_epoch % trainer.val_iter == 0:
            trainer.eval(dataloaders['val'])
    trainer.test(dataloaders['test'])
    
    # ###########
    # # define model
    # net = define_model(opts)
    
    # ###########
    # # define other hyperparameters
    # loss = define_loss()
    # optimizer = define_optimizer()
    # scheduler = define_scheduler()
    # scaler = define_scaler()
    # precision = torch.float32
    
    # ###########
    # # load previously trained state_dict
    # curr_epoch = 0
    # ###########
    # # Train
    # for epoch in range(curr_epoch, max_epochs):
    #     for batch in dataloaders['train']:
    #         with torch.cuda.amp.autocast(precision):
    #             x = batch['source'].to(device)
    #             y = batch['target'].to(device)
    #             out = net(x)
    #             loss = loss(out, y)
    #             optimizer.zero_grad()
    #             if scaler is not None:
    #                 scaler.scale(loss).backward()
    #                 scaler.update(1)
    #                 scaler.step()
    #             else:
    #                 optimizer.step()
    #     scheduler.step()
    #     # report progression
    #     out = out.float()
            
    #     # run for evaluation
    #     if curr_epoch % eval_iter == 0:
    #         with torch.no_grad():
    #             for batch in dataloaders['val']:
    #                 x = batch['source'].to(device)
    #                 y = batch['target'].to(device)
    #                 out = net(x)
    #                 loss = loss(out, y)
    #                 # calculate metrics
    #                 metrics
    #         # report progression
    #         # save best state_dict
    #     # collect record
    #     # save latest state_dict
    # # ###########
    # # # run final test
    # # with torch.no_grad():
    # #     for batch in dataloaders['test']:
            

if __name__ == '__main__':
    main()