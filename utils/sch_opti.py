from torch import optim
def getoptimizer(optimizername,cfg,net):
    if optimizername == 'Adam':
        optimizer = optim.Adam(net.parameters(),
                               lr=cfg.lr)
    elif optimizername == 'RMSprop':
        optimizer = optim.RMSprop(net.parameters(),
                                  lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    else:
        optimizer = optim.SGD(net.parameters(),
                              lr=cfg.lr,
                              momentum=cfg.momentum,
                              weight_decay=cfg.weight_decay,
                              nesterov=cfg.nesterov)#Nesterov Momentum
    return optimizer
def getscheduler(schedulername,cfg,optimizer):
    if schedulername=='MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg.lr_decay_milestones,
                                               gamma=cfg.lr_decay_gamma)
    elif schedulername=='ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.35, verbose=1, min_lr=0.0001,
                                                           patience=125)
    return scheduler