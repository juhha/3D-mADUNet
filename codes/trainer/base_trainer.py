import os

class BaseTrainer:
    def __init__(
        self,
        opts,
        args,
    ):
        super().__init__()
        self.opts = opts
        self.args = args
        self.fold = args.fold
        # self.cp_dir = os.path.join(base_dir, 'checkpoint', args.config_file) # make checkpoint directory
        self.cp_dir = opts['checkpoint']
        self.device = args.device # set device
        self.monitor_metrics = opts['train_opt']['monitor_metrics']
        self.max_epoch = opts['train_opt']['max_epoch']
        self.val_iter = opts['train_opt']['val_iter']
        self.source_key = opts['train_opt']['source_key']
        self.target_key = opts['train_opt']['target_key']
        # initialize network, loss, optimizer, scheduler...
        self.initialize_train()
    def initialize_train(self):
        pass
    def train(self, dataloaders):
        pass
    def train_iter(self, batch):
        pass
    def eval(self, dl):
        pass
    def test(self, dl):
        pass
    def save_checkpoint(self, save_fname = 'latest.pt'):
        pass