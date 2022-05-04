import warnings
import sys

warnings.filterwarnings("ignore")
sys.path.append('..')

import time
from tqdm import tqdm
import os
import argparse

import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.cuda import amp

from network import EfficientUNet, UperNet, get_smp_model
from utils.losses import SoftDiceLoss, BCEDicedLoss, get_losses, BCETverskyLoss, BCEDicedLossV2
from data.dataset import BuildDataset, read_csv, prepare_loaders
from utils.utils import AverageMeter, Logger, calculate_dice, calculate_metric_score, dice_coef


def train_epoch(epoch, data_set, model, criterion, optimizer, logger):
    print('train at epoch {}'.format(epoch))

    model.train()

    losses = AverageMeter()
    dices = AverageMeter()
    accs = AverageMeter()
    aucs = AverageMeter()

    train_loader = DataLoader(dataset=data_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    training_process = tqdm(train_loader)
    for i, (inputs, targets) in enumerate(training_process):
        if i > 0:
            training_process.set_description("Epoch %d -- Loss: %.4f, Dice: %.4f" %
                                             (epoch, losses.avg.item(), dices.avg.item()))

        inputs = inputs.type(torch.FloatTensor)
        inputs = inputs.cuda(device_id)
        targets = targets.type(torch.FloatTensor)
        targets = targets.cuda(device_id)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        if activation is None:
            outputs = torch.nn.Sigmoid()(outputs)
        dice = dice_coef(y_pred=outputs.cpu().detach(), y_true=targets.cpu().detach())
        # dice = calculate_dice(outputs.cpu().detach(), targets.cpu().detach())

        losses.update(loss.cpu().detach(), inputs.size(0))
        dices.update(dice, inputs.size(0))
        # accs.update(acc, inputs.size(0))
        # aucs.update(auc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Train:\t Loss:{0:.4f}\t Dice:{1:.4f}\t".format(losses.avg, dices.avg))

    logger.log(phase="train", values={
        'epoch': epoch,
        'loss': format(losses.avg.item(), '.4f'),
        'dice': format(dices.avg.item(), '.4f'),
        'f1': format(0.0, '.4f'),
        'iou': format(0.0, '.4f'),
        'lr': optimizer.param_groups[0]['lr']
    })

    return dices.avg


def train_epoch_amp(epoch, data_set, model, criterion, optimizer, logger):
    print('train at epoch {}'.format(epoch))

    model.train()
    scaler = amp.GradScaler()

    losses = AverageMeter()
    dices = AverageMeter()
    accs = AverageMeter()
    aucs = AverageMeter()

    train_loader = DataLoader(dataset=data_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    training_process = tqdm(train_loader)
    for i, (inputs, targets) in enumerate(training_process):
        if i > 0:
            training_process.set_description("Epoch %d -- Loss: %.4f, Dice: %.4f" %
                                             (epoch, losses.avg.item(), dices.avg.item()))

        inputs = inputs.type(torch.FloatTensor)
        inputs = inputs.cuda(device_id)
        targets = targets.type(torch.FloatTensor)
        targets = targets.cuda(device_id)

        with amp.autocast(enabled=True):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss = loss / n_accumulate

        scaler.scale(loss).backward()

        if (i + 1) % n_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

        if activation is None:
            outputs = torch.nn.Sigmoid()(outputs)
        dice = dice_coef(y_pred=outputs.cpu().detach(), y_true=targets.cpu().detach())
        # dice = calculate_dice(outputs.cpu().detach(), targets.cpu().detach())

        losses.update(loss.cpu().detach(), inputs.size(0))
        dices.update(dice, inputs.size(0))
        # accs.update(acc, inputs.size(0))
        # aucs.update(auc, inputs.size(0))

    print("Train:\t Loss:{0:.4f}\t Dice:{1:.4f}\t LR {2:.4f}".
          format(losses.avg, dices.avg, optimizer.param_groups[0]['lr']))

    logger.log(phase="train", values={
        'epoch': epoch,
        'loss': format(losses.avg.item(), '.4f'),
        'dice': format(dices.avg.item(), '.4f'),
        'f1': format(0.0, '.4f'),
        'iou': format(0.0, '.4f'),
        'lr': optimizer.param_groups[0]['lr']
    })

    return dices.avg


def val_epoch(epoch, data_set, model, criterion, optimizer, logger=None):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    losses = AverageMeter()
    dices = AverageMeter()
    accs = AverageMeter()
    f1s = AverageMeter()
    ious = AverageMeter()

    valildation_loader = DataLoader(dataset=data_set,
                                    batch_size=val_batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)
    val_process = tqdm(valildation_loader)
    start_time = time.time()
    for i, (inputs, targets) in enumerate(val_process):
        if i > 0:
            val_process.set_description("Epoch %d -- Loss: %.4f, Dice: %.4f, F1: %.4f, IoU: %.4f" %
                                        (epoch, losses.avg.item(), dices.avg.item(), f1s.avg, ious.avg))

        inputs = inputs.type(torch.FloatTensor)
        inputs = inputs.cuda(device_id)
        targets = targets.type(torch.FloatTensor)
        targets = targets.cuda(device_id)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # dice = calculate_dice(outputs.cpu().detach(), targets.cpu().detach())
        if activation is None:
            outputs = torch.nn.Sigmoid()(outputs)
        dice = dice_coef(y_pred=outputs.cpu().detach(), y_true=targets.cpu().detach(), thr=0.5)
        acc, f1, iou = calculate_metric_score(outputs.cpu().detach().numpy(), targets.cpu().detach().numpy(),
                                              metric_name=None)

        losses.update(loss.cpu(), inputs.size(0))
        dices.update(dice, inputs.size(0))
        accs.update(acc, inputs.size(0))
        f1s.update(f1, inputs.size(0))
        ious.update(iou, inputs.size(0))

    epoch_time = time.time() - start_time

    print("Test:\t Dice:{0:.4f} \t F1:{1:.4f} \t IoU:{2:.4f} \t using:{3:.3f} minutes".
          format(dices.avg, f1s.avg, ious.avg, epoch_time / 60))
    if logger is not None:
        logger.log(phase="val", values={
            'epoch': epoch,
            'loss': format(losses.avg.item(), '.4f'),
            'dice': format(dices.avg.item(), '.4f'),
            'f1': format(f1s.avg.item(), '.4f'),
            'iou': format(ious.avg.item(), '.4f'),
            'lr': optimizer.param_groups[0]['lr']
        })

    return losses.avg, dices.avg


parser = argparse.ArgumentParser(description="UWMGI  @cby Training")
parser.add_argument("--device_id", default=0, help="Setting the GPU id", type=int)
parser.add_argument("--batch_size", default=64, help="Setting batch_size", type=int)
parser.add_argument("--k", default=0, help="The value of K Fold", type=int)
parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()

if __name__ == '__main__':
    csv_path = '/data1/chenby/dataset/UWMGI/train_mask.csv'
    activation = None  # None torch.nn.Sigmoid()
    fold = args.k
    img_size = 266  # [h,w] 256
    start_epoch = 1
    epochs = start_epoch + 50
    device_id = args.device_id
    lr = 1e-4  # 1e-3 , convnext 1e-4
    batch_size = args.batch_size  # 64
    val_batch_size = batch_size
    num_workers = 4
    use_amp = True
    n_accumulate = 2

    model_name = f'upernet_convnext_l_{img_size}'
    # model_name = f'upernet_efn_b6_{img_size}'
    # model_name = f'smp_unet_efn_b6_{img_size}'

    if use_amp:
        model_name += '_amp'
    write_file = f'../output/logs/{model_name}/fold{args.k}'
    save_dir = f'../output/weights/{model_name}/fold{args.k}'

    # model = EfficientUNet(model_name='efficientnet-b4', n_channels=1, n_classes=3, activation=torch.nn.Sigmoid())
    # model = get_smp_model(model_name='efficientnet-b6', num_classes=3, in_channels=3,
    #                       activation=activation, model_type='unet')
    model = UperNet(backbone='convnext_large_22k', num_classes=3, in_channels=3,
                    activation=activation)  # tf_efficientnetv2_s_in21ft1k convnext_base_22k
    # print(model)
    model_path = None
    # model_path = '/data1/chenby/py_project/UWMGI/output/weights/smp_unet_efn_b6_256_amp/epoch_13_0.8932.pth'
    if model_path is not None:
        # model = torch.load(model_path)
        model.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'])
        print('Model found in {}'.format(model_path))
    else:
        print('No model found, initializing random model.')
    model = model.cuda(device_id)
    # print(model)
    # loss_function = SoftDiceLoss()
    # loss_function = torch.nn.BCELoss()
    # loss_function = BCEDicedLoss(bce_weight=0.5, dice_weight=0.5)  # 0.1
    # loss_function = BCEDicedLossV2(bce_weight=0.5, dice_weight=0.5)  # 0.1

    loss_function = BCETverskyLoss(bce_weight=0.5, tversky_weight=0.5)
    loss_function = loss_function.cuda(device_id)
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=lr, weight_decay=1e-5)
    # optimizer = optim.SGD(parameters, lr=lr, weight_decay=1e-5)
    if model_path is not None:
        optimizer.load_state_dict(torch.load(model_path, map_location='cpu')['optimizer'])

    # dataset
    df = read_csv(csv_path=csv_path)
    training_data, validation_data = prepare_loaders(df, fold, batch_size=batch_size,
                                                     img_size=img_size, return_dataset=True)
    is_train = True
    if is_train:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        train_logger = Logger(model_name=write_file, header=['epoch', 'loss', 'dice', 'f1', 'iou', 'lr'])
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.6, patience=2)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6, last_epoch=-1)
        max_val_acc = 0.
        for i in range(start_epoch, epochs):
            if not use_amp:
                train_epoch(epoch=i,
                            data_set=training_data,
                            model=model,
                            criterion=loss_function,
                            optimizer=optimizer,
                            logger=train_logger)
            else:
                train_epoch_amp(epoch=i,
                                data_set=training_data,
                                model=model,
                                criterion=loss_function,
                                optimizer=optimizer,
                                logger=train_logger)
            val_loss, val_acc = val_epoch(epoch=i,
                                          data_set=validation_data,
                                          model=model,
                                          criterion=loss_function,
                                          optimizer=optimizer,
                                          logger=train_logger)
            scheduler.step(val_loss)
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_states_path = os.path.join(save_dir, 'epoch_{0}_{1:.4f}.pth'.format(i, val_acc))
                states = {
                    'epoch': i + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(states, save_states_path)
            print('Current acc:', val_acc, 'Best acc:', max_val_acc)

        save_model_path = os.path.join(save_dir, "last_model_file" + str(epochs) + ".pth")
        if os.path.exists(save_model_path):
            os.system("rm " + save_model_path)
        torch.save(states, save_model_path)
    else:
        val_loss, val_acc = val_epoch(epoch=0,
                                      data_set=validation_data,
                                      model=model,
                                      criterion=loss_function,
                                      optimizer=optimizer,
                                      logger=None)
