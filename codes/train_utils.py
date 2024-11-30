import torch

class DummyGradScaler:
    def __init__(self):
        pass

    def scale(self, loss):
        # This function takes the loss and returns it without any modification
        return loss

    def step(self, optimizer):
        # This function simply calls the step function of the optimizer
        optimizer.step()

    def update(self):
        # This function is a placeholder and does nothing in the dummy scaler
        pass

    def unscale_(self, optimizer):
        # This function is a placeholder and does nothing in the dummy scaler
        pass
    def load_state_dict(self, state_dict):
        pass
    def state_dict(self):
        return {}

class DummyScheduler:
    def __init__(self, optimizer):
        pass
    def state_dict(self):
        return {}
    def step(self):
        pass        

def build_optimizer(params, optim_opt):
    optim_type = optim_opt['type']
    optim_params = optim_opt['params']
    if optim_type == 'adam':
        return torch.optim.Adam(params, **optim_params)

def build_scheduler(optimizer, scheduler_opt):
    if scheduler_opt is None:
        return DummyScheduler(optimizer)
    scheduler_type = scheduler_opt['type']
    scheduler_params = scheduler_opt['params']
    if scheduler_type == 'multisteplr':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_params)
    
def build_scaler(use_scaler, params):
    if not use_scaler:
        return DummyGradScaler()
    else:
        return torch.cuda.amp.GradScaler(**params)