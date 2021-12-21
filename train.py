import argparse
from PIL import Image
from torchvision import transforms

import cv2
import logging
import os
import os.path as osp
import sys
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import strftime, localtime
from torch.utils.data import DataLoader, random_split
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


from tqdm import tqdm
from pandas import DataFrame

from utils.pre_add_gt import confusion_image
from utils.config import Config
from utils.losses import LovaszLossSoftmax, LovaszLossHinge,DiceLoss,LogNLLLoss,structure_loss
from torch.nn.modules.loss import CrossEntropyLoss
from utils.judge import dice,iou
from utils.evaluate import eval_net,test_net
from utils.sch_opti import getscheduler,getoptimizer
from utils.dataset import ImageToImage2D
from utils.transformation import JointTransform2D
from torch.utils.tensorboard import SummaryWriter
from networkarchitest.interaction import LTUNet
from torchsummary import summary
cfg = Config()

def train_net(net, cfg, save_path):
    # ------------------------open file，make dataset
    crop = None
    tf_train = JointTransform2D(crop=crop, p_flip=0.5, color_jitter_params=None, long_mask=True)
    tf_val = JointTransform2D(crop=crop, p_flip=0, color_jitter_params=None, long_mask=True)
    dataset =ImageToImage2D(cfg.images_dir,tf_train)
    testset = ImageToImage2D(cfg.testimages_dir,tf_val)
    wrtrain=[]
    wrtest=[]
    wrtotal=[]
    #------------------------open file，make dataset
    best_score = 0
    val_percent = cfg.validation / 100
    n_val =int(val_percent*len(dataset)) #int(val_percent*len(dataset)) #len(testset)#
    n_train = len(dataset)-n_val
    train, val = random_split(dataset, [n_train, n_val])
    #train=dataset
    train_loader = DataLoader(train,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True)
    val_loader = DataLoader(val,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True)
    test_loader = DataLoader(testset,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True)

    writer = SummaryWriter(comment=f'MODEL_{cfg.model}_LR_{cfg.lr}_BS_{cfg.batch_size}_SCALE_{cfg.scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {cfg.epochs}
        Batch size:      {cfg.batch_size}
        Learning rate:   {cfg.lr}
        Optimizer:       {cfg.optimizer}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {cfg.save_cp}
        Device:          {device.type}
        Images scaling:  {cfg.scale}
    ''')

    optimizer=getoptimizer(cfg.optimizer,cfg,net)
    scheduler = getscheduler('MultiStepLR',cfg,optimizer)

    if cfg.n_classes > 1:
        criterion = LovaszLossSoftmax()

    else:
        criterion = LovaszLossSoftmax()

    for epoch in range(cfg.epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{cfg.epochs}', unit='img') as pbar:
            for batch in train_loader:
                batch_imgs = batch['image']
                batch_masks = batch['mask']
                assert batch_imgs.shape[1] == cfg.n_channels, \
                        f'Network has been defined with {cfg.n_channels} input channels, ' \
                        f'but loaded images have {batch_imgs.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                batch_imgs = batch_imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if cfg.n_classes == 1 else torch.long
                batch_masks = batch_masks.to(device=device, dtype=mask_type)

                inference_masks = net(batch_imgs)

                if cfg.n_classes == 1:
                    inferences = inference_masks
                    masks = batch_masks
                else:
                    inferences = inference_masks
                    masks = batch_masks


                loss=structure_loss(inferences, masks.unsqueeze(1))
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('model/lr', optimizer.param_groups[0]['lr'], global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                pbar.update(batch_imgs.shape[0])
                global_step += 1


        #--------------------------------------------------validation
        val_score,iou_score,dice_score = eval_net(net, val_loader, device, n_val, cfg)
        if cfg.n_classes > 1:
            logging.info('Validation cross entropy: {}'.format(val_score))
            writer.add_scalar('CrossEntropy/test', val_score, global_step)
        else:
            logging.info('Validation Dice Coeff: {}'.format(val_score))
            writer.add_scalar('Dice/test', val_score, global_step)

        writer.add_images('images', np.array(batch_imgs.cpu(), dtype='uint8'), global_step)
        if cfg.deepsupervision:
            inference_masks = inference_masks[-1]
        if cfg.n_classes == 1:
            inference_mask = torch.sigmoid(inference_masks) > cfg.out_threshold
            #print(batch_imgs.shape,inference_mask.shape,batch_masks.shape)
            output,_=confusion_image(batch_imgs,inference_mask,batch_masks)
            writer.add_images('masks/inference',
                               toTensor(output),
                               global_step)

        else:
            inference_masks= F.softmax(inference_masks,dim=1)
            ids = inference_masks.shape[1]  # N x C x H x W
            inference_masks = torch.chunk(inference_masks, ids, dim=1)
            for idx in range(0, len(inference_masks)):
                inference_mask =inference_masks[idx] > cfg.out_threshold
                writer.add_images('masks/inference_' + str(idx),
                                 inference_mask[idx],
                                  global_step)
        #--------------------------test-------------------------

        #--------------------------test-------------------------
        if val_score>best_score:
            best_score=val_score
            if cfg.save_cp:
                try:
                    os.mkdir(cfg.checkpoints_dir)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass

                ckpt_name = 'dicebest.pth'
                torch.save(net.state_dict(),
                           os.path.join(save_path,"best_score.pth"))
                logging.info(f'Checkpoint {epoch + 1} saved !')

        #日志写入CSV文件中------------------------------------------------------------
        wrtrain.append( ''.join(str(i) for i in [epoch,':',epoch_loss]))
        wrtest.append(''.join(str(i) for i in [epoch, ':', val_score," miou:",iou_score," dice:",dice_score]))

    res1,res2,res3=test_net(net, test_loader, device, len(testset), cfg, save_path)
    wrtotal.append(''.join(str(i) for i in [cfg.model, ":"," miou:", res2, " dice:", res3]))
    resultdataframe = {"trainloss": wrtrain,
                      "validation": wrtest,
                       "test": wrtotal}#
    resultdataframe  = DataFrame.from_dict(resultdataframe,orient='index' ).transpose()
    resultdataframe.to_csv(os.path.join(save_path,'Result.csv'), mode='w')
    torch.cuda.empty_cache()


def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    img=img.transpose(0, 3, 1, 2)
    img=np.array(img, dtype='uint8')
    return img # 255也可以改为256





if __name__ == '__main__':
    # create folders
    now_time = strftime('%Y-%m-%d %H:%M:%S', localtime())
    now_time = now_time.split(' ')
    network='LTUNet'


    #start
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    print(torch.cuda.is_available())

    try:
            if not(os.path.isdir('./train_log/')):
                os.mkdir('./train_log/')
            trainlog_path = "./train_log/" + now_time[0] + '_' + now_time[1].split(':')[0] + '_' + \
                            now_time[1].split(':')[1] + '_' + now_time[1].split(':')[2] + str(network)
            os.makedirs(trainlog_path)

            net = eval(network)(3,cfg.n_classes,inner_dim=192,patches=[5,5])
            logging.info(f'Network:\n'
                         f'\t{network} model\n'
                         f'\t{cfg.n_channels} input channels\n'
                         f'\t{cfg.n_classes} output channels (classes)\n'
                         f'\t{"Bilinear" if cfg.bilinear else "Dilated conv"} upscaling')

            if cfg.load:
                net.load_state_dict(
                    torch.load("./", map_location=device)
                )
                logging.info(f'Model loaded from {cfg.load}')

            net.to(device=device)
            train_thistime_path=trainlog_path+'/time'+str(0)
            os.makedirs(train_thistime_path)

            train_net(net=net, cfg=cfg,save_path=train_thistime_path)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
