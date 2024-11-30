from .srgan_trainer import SRGANTrainer

def define_trainer(args, opts):
    trainer_opt = opts['trainer_opt']
    trainer_type = trainer_opt['type']
    if trainer_type == 'srgan':
        return SRGANTrainer(opts, args)