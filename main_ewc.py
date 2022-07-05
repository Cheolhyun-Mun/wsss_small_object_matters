#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   07 January 2019

from __future__ import absolute_import, division, print_function

import json
import multiprocessing
import os

import click
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchnet.meter import MovingAverageValueMeter
from tqdm import tqdm

from libs.datasets import get_dataset
from libs.models import DeepLabV2_ResNet101_MSC
from libs.utils import DenseCRF, PolynomialLR, mIoU, IAmIoU
from libs.utils import get_regularizer
from libs.utils import CELoss

import random


def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        print("Device:")
        for i in range(torch.cuda.device_count()):
            print("    {}:".format(i), torch.cuda.get_device_name(i))
    else:
        print("Device: CPU")
    return device


def get_params(model, key):
    # For Dilated FCN
    if key == "1x":
        for m in model.named_modules():
            if "layer" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        yield p
    # For conv weight in the ASPP module
    if key == "10x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].weight
    # For conv bias in the ASPP module
    if key == "20x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].bias


def resize_labels(labels, size):
    """
    Downsample labels for 0.5x and 0.75x logits by nearest interpolation.
    Other nearest methods result in misaligned labels.
    -> F.interpolate(labels, shape, mode='nearest')
    -> cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
    """
    new_labels = []
    for label in labels:
        label = label.float().numpy()
        label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
        new_labels.append(np.asarray(label))
    new_labels = torch.LongTensor(new_labels)
    return new_labels

def set_random_seed(seed):
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

@click.group()
@click.pass_context
def main(ctx):
    """
    Training and evaluation
    """
    print("Mode:", ctx.invoked_subcommand)


@main.command()
@click.option(
    "-c",
    "--config-path",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "--margin",
    required=True,
    help="margin_num",
)
@click.option(
    "--ewc_reg",
    default=500,
    help="ewc_reg",
)
@click.option(
    "--ewc_iter",
    default=1000,
    help="ewc_iter",
)
@click.option(
    "--cuda/--cpu", default=True, help="Enable CUDA if available [default: --cuda]"
)
def train(config_path, margin, ewc_reg, ewc_iter, cuda):
    """
    Training DeepLab by v2 protocol
    """

    set_random_seed(100)

    # Configuration
    CONFIG = OmegaConf.load(config_path)
    device = get_device(cuda)
    torch.backends.cudnn.benchmark = True

    # Model check
    print("Model:", CONFIG.MODEL.NAME)
    assert (
        CONFIG.MODEL.NAME == "DeepLabV2_ResNet101_MSC"
    ), 'Currently support only "DeepLabV2_ResNet101_MSC"'

    # Model setup
    model = DeepLabV2_ResNet101_MSC(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load(CONFIG.MODEL.INIT_MODEL)
    print("    Init:", CONFIG.MODEL.INIT_MODEL)
    for m in model.base.state_dict().keys():
        if m not in state_dict.keys():
            print("    Skip init:", m)
    model.base.load_state_dict(state_dict, strict=False)  # to skip ASPP
    model = nn.DataParallel(model)
    model.to(device)

    # Path to save models
    checkpoint_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "models",
        CONFIG.EXP.ID,
        f"{margin}_{ewc_reg}_{ewc_iter}",
        CONFIG.MODEL.NAME.lower(),
        CONFIG.DATASET.SPLIT.TRAIN,
    )
    makedirs(checkpoint_dir)
    print("Checkpoint dst:", checkpoint_dir)

    # Freeze the batch norm pre-trained on COCO
    model.train()
    model.module.base.freeze_bn()

    # Dataset
    dataset = get_dataset(CONFIG.DATASET.NAME)(
        root=CONFIG.DATASET.ROOT,
        split=CONFIG.DATASET.SPLIT.TRAIN,
        ignore_label=CONFIG.DATASET.IGNORE_LABEL,
        mean_bgr=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        augment=True,
        base_size=CONFIG.IMAGE.SIZE.BASE,
        crop_size=CONFIG.IMAGE.SIZE.TRAIN,
        scales=CONFIG.DATASET.SCALES,
        flip=True,
    )
    print(dataset)

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CONFIG.SOLVER.BATCH_SIZE.TRAIN,
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,
        shuffle=True,
    )
    loader_iter = iter(loader)

    # Optimizer
    optimizer = torch.optim.SGD(
        # cf lr_mult and decay_mult in train.prototxt
        params=[
            {
                "params": get_params(model.module, key="1x"),
                "lr": CONFIG.SOLVER.LR,
                "weight_decay": CONFIG.SOLVER.WEIGHT_DECAY,
            },
            {
                "params": get_params(model.module, key="10x"),
                "lr": 10 * CONFIG.SOLVER.LR,
                "weight_decay": CONFIG.SOLVER.WEIGHT_DECAY,
            },
            {
                "params": get_params(model.module, key="20x"),
                "lr": 20 * CONFIG.SOLVER.LR,
                "weight_decay": 0.0,
            },
        ],
        momentum=CONFIG.SOLVER.MOMENTUM,
    )

    # Learning rate scheduler
    scheduler = PolynomialLR(
        optimizer=optimizer,
        step_size=CONFIG.SOLVER.LR_DECAY,
        iter_max=CONFIG.SOLVER.ITER_MAX,
        power=CONFIG.SOLVER.POLY_POWER,
    )

    flag = False

 
    # Setup loss logger
    writer = SummaryWriter(os.path.join(CONFIG.EXP.OUTPUT_DIR, "logs", CONFIG.EXP.ID))
    average_loss = MovingAverageValueMeter(CONFIG.SOLVER.AVERAGE_LOSS)

    model_old = None
    old_state = None

    #regularizer state
    regularizer = get_regularizer(model, model_old, device, old_state)
    reg_importance = ewc_reg

    for iteration in tqdm(
        range(1, CONFIG.SOLVER.ITER_MAX + 1),
        total=CONFIG.SOLVER.ITER_MAX,
        dynamic_ncols=True,
    ):

        if flag and model_old is None:
            print(
                '=> Estimating diagonals of the fisher information matrix...',
                flush=True, end='',
            )
            model_old = DeepLabV2_ResNet101_MSC(n_classes=CONFIG.DATASET.N_CLASSES)
            state_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint_final_old.ckpt"))
            print("Init:", os.path.join(checkpoint_dir, "checkpoint_final_old.ckpt"),flush=True, end='')
            for m in model_old.state_dict().keys():
                if m not in state_dict["model_state"].keys():
                    print("    Skip init:", m)
            model_old.load_state_dict(state_dict["model_state"])  # to skip ASPP
            #model_old = nn.DataParallel(model_old)
            #model_old.to(device)
            for par in model_old.parameters():
                par.requires_grad = False
            
            old_state = state_dict["regularizer_state"]

            model_old.eval()
            regularizer = get_regularizer(model, model_old, device, old_state)



        # Clear gradients (ready to accumulate)
        optimizer.zero_grad()

        loss = 0
        for _ in range(CONFIG.SOLVER.ITER_SIZE):
            try:
                _, images, labels = next(loader_iter)
            except:
                loader_iter = iter(loader)
                _, images, labels = next(loader_iter)

            # Propagate forward
            logits = model(images.to(device))

            # Loss
            iter_loss = 0
            for logit in logits:
                # Resize labels for {100%, 75%, 50%, Max} logits
                _, _, H, W = logit.shape
                labels_ = resize_labels(labels, size=(H, W))
                iter_loss += CELoss(logit, labels_.to(device), margin=int(margin), give_weight=flag)

            # Propagate backward (just compute gradients)
            iter_loss /= CONFIG.SOLVER.ITER_SIZE
            iter_loss.backward()

            regularizer.update()
            l_reg = reg_importance * regularizer.penalty()
            if l_reg != 0.:
                l_reg.backward()

            loss += float(iter_loss)

            

        average_loss.add(loss)

        # Update weights with accumulated gradients
        optimizer.step()

        # Update learning rate
        scheduler.step(epoch=iteration)

        # TensorBoard
        if iteration % CONFIG.SOLVER.ITER_TB == 0:
            writer.add_scalar("loss/train", average_loss.value()[0], iteration)
            #print("loss/train", average_loss.value()[0], iteration)
            for i, o in enumerate(optimizer.param_groups):
                writer.add_scalar("lr/group_{}".format(i), o["lr"], iteration)
            for i in range(torch.cuda.device_count()):
                writer.add_scalar(
                    "gpu/device_{}/memory_cached".format(i),
                    torch.cuda.memory_cached(i) / 1024 ** 3,
                    iteration,
                )

        # Save a model
        if iteration % CONFIG.SOLVER.ITER_SAVE == 0:
            torch.save(
                model.module.state_dict(),
                os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(iteration)),
            )
        
        if not flag and iteration % int(ewc_iter) == 0:
            torch.save(
                {"model_state": model.module.state_dict(),
                "regularizer_state": regularizer.state_dict()
                }, 
                os.path.join(checkpoint_dir, "checkpoint_final_old.ckpt")
            )
            flag = True
    torch.save(
        model.module.state_dict(), os.path.join(checkpoint_dir, "checkpoint_final.pth")
    )


@main.command()
@click.option(
    "-c",
    "--config-path",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "--margin",
    required=True,
    help="margin_num",
)
@click.option(
    "--ewc_reg",
    default=500,
    help="ewc_reg",
)
@click.option(
    "--ewc_iter",
    default=1000,
    help="ewc_iter",
)
@click.option(
    "--cuda/--cpu", default=True, help="Enable CUDA if available [default: --cuda]"
)
def test(config_path, margin, ewc_reg, ewc_iter, cuda):
    """
    Evaluation on validation set
    """

    # Configuration
    CONFIG = OmegaConf.load(config_path)
    device = get_device(cuda)
    torch.set_grad_enabled(False)

    # Dataset
    dataset = get_dataset(CONFIG.DATASET.NAME)(
        root=CONFIG.DATASET.ROOT,
        split=CONFIG.DATASET.SPLIT.VAL,
        ignore_label=CONFIG.DATASET.IGNORE_LABEL,
        mean_bgr=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        augment=False,
    )
    print(dataset)

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CONFIG.SOLVER.BATCH_SIZE.TEST,
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,
        shuffle=False,
    )

    model_path = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "models",
        CONFIG.EXP.ID,
        f"{margin}_{ewc_reg}_{ewc_iter}",
        CONFIG.MODEL.NAME.lower(),
        CONFIG.DATASET.SPLIT.TRAIN,
        "checkpoint_final.pth"
    )

    # Model
    model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model = nn.DataParallel(model)
    model.eval()
    model.to(device)

    # Path to save logits
    logit_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "features",
        CONFIG.EXP.ID,
        f"{margin}_{ewc_reg}_{ewc_iter}",
        CONFIG.MODEL.NAME.lower(),
        CONFIG.DATASET.SPLIT.VAL,
        "logit",
    )
    makedirs(logit_dir)
    print("Logit dst:", logit_dir)

    # Path to save scores
    save_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "scores",
        CONFIG.EXP.ID,
        f"{margin}_{ewc_reg}_{ewc_iter}",
        CONFIG.MODEL.NAME.lower(),
        CONFIG.DATASET.SPLIT.VAL,
    )
    makedirs(save_dir)
    save_path = os.path.join(save_dir, "scores.json")
    print("Score dst:", save_path)
    save_path_IAmIoU = os.path.join(save_dir, "scores_IAmIoU.json")
    print("Score dst:", save_path)

    iu_stack = np.zeros(CONFIG.DATASET.N_CLASSES)
    class_stack = np.zeros(CONFIG.DATASET.N_CLASSES)
    iu_small_stack = np.zeros(CONFIG.DATASET.N_CLASSES)
    class_small_stack = np.zeros(CONFIG.DATASET.N_CLASSES)
    iu_medium_stack = np.zeros(CONFIG.DATASET.N_CLASSES)
    class_medium_stack = np.zeros(CONFIG.DATASET.N_CLASSES)
    iu_large_stack = np.zeros(CONFIG.DATASET.N_CLASSES)
    class_large_stack = np.zeros(CONFIG.DATASET.N_CLASSES)
    hist = np.zeros((CONFIG.DATASET.N_CLASSES, CONFIG.DATASET.N_CLASSES))
    preds, gts, ins_gts = [], [], []
    
    for image_ids, images, gt_labels, gt_inst in tqdm(
        loader, total=len(loader), dynamic_ncols=True
    ):
        # Image
        images = images.to(device)

        # Forward propagation
        logits = model(images)

        # Save on disk for CRF post-processing
        for image_id, logit in zip(image_ids, logits):
            filename = os.path.join(logit_dir, image_id + ".npy")
            np.save(filename, logit.cpu().numpy())

        # Pixel-wise labeling
        _, H, W = gt_labels.shape
        logits = F.interpolate(
            logits, size=(H, W), mode="bilinear", align_corners=False
        )
        probs = F.softmax(logits, dim=1)
        labels = torch.argmax(probs, dim=1)

        preds = list(labels.cpu().numpy())
        gts = list(gt_labels.numpy())
        ins_gts = list(gt_inst.numpy())

        # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
        result_IAmIoU = IAmIoU(ins_gts, gts, preds, n_class=CONFIG.DATASET.N_CLASSES)
        result_miou = mIoU(gts, preds, n_class=CONFIG.DATASET.N_CLASSES)

        iu_stack += result_IAmIoU["IoU stack"]
        class_stack += result_IAmIoU["Class stack"]
        iu_small_stack += result_IAmIoU["Small IoU stack"]
        class_small_stack += result_IAmIoU["Small Class stack"]
        iu_medium_stack += result_IAmIoU["Medium IoU stack"]
        class_medium_stack += result_IAmIoU["Medium Class stack"]
        iu_large_stack += result_IAmIoU["Large IoU stack"]
        class_large_stack += result_IAmIoU["Large Class stack"]
        hist += result_miou["hist"]

    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(CONFIG.DATASET.N_CLASSES), iu))

    iu_stack /= class_stack
    iu_mean = np.nanmean(iu_stack)
    cls_iu_mean = dict(zip(range(CONFIG.DATASET.N_CLASSES), iu_stack))

    iu_small_stack /= class_small_stack
    small_iu_mean = np.nanmean(iu_small_stack)
    small_cls_iu_mean = dict(zip(range(CONFIG.DATASET.N_CLASSES), iu_small_stack))

    iu_medium_stack /= class_medium_stack
    medium_iu_mean = np.nanmean(iu_medium_stack)
    medium_cls_iu_mean = dict(zip(range(CONFIG.DATASET.N_CLASSES), iu_medium_stack))

    iu_large_stack /= class_large_stack
    large_iu_mean = np.nanmean(iu_large_stack)
    large_cls_iu_mean = dict(zip(range(CONFIG.DATASET.N_CLASSES), iu_large_stack))



    score_IAmIoU = {
         "Mean IoU": iu_mean,
         "Class IoU": cls_iu_mean,
         "Small Mean IoU": small_iu_mean,
         "Small Class IoU": small_cls_iu_mean,
         "Medium Mean IoU": medium_iu_mean,
         "Medium Class IoU": medium_cls_iu_mean,
         "Learge Mean IoU": large_iu_mean,
         "Large Class IoU": large_cls_iu_mean,
     }
    score = {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }
        
    with open(save_path, "w") as f:
        json.dump(score, f, indent=4, sort_keys=True)
    with open(save_path_IAmIoU, "w") as g:
        json.dump(score_IAmIoU, g, indent=4, sort_keys=True)

@main.command()
@click.option(
    "-c",
    "--config-path",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "--margin",
    required=True,
    help="margin_num",
)
@click.option(
    "--ewc_reg",
    default=500,
    help="ewc_reg",
)
@click.option(
    "--ewc_iter",
    default=1000,
    help="ewc_iter",
)
@click.option(
    "--data_type",
    default=True,
    help="ewc_iter",
)
@click.option(
    "-j",
    "--n-jobs",
    type=int,
    default=multiprocessing.cpu_count()/2,
    show_default=True,
    help="Number of parallel jobs",
)
def crf(config_path, margin, ewc_reg, ewc_iter, data_type, n_jobs):
    """
    CRF post-processing on pre-computed logits
    """

    # Configuration
    CONFIG = OmegaConf.load(config_path)
    torch.set_grad_enabled(False)
    print("# jobs:", n_jobs)

    data_path = "voc"
    if not data_type:
        data_path = "our"
        CONFIG.DATASET.SPLIT.VAL = "val_our"

    # Dataset
    dataset = get_dataset(CONFIG.DATASET.NAME)(
        root=CONFIG.DATASET.ROOT,
        split=CONFIG.DATASET.SPLIT.VAL,
        ignore_label=CONFIG.DATASET.IGNORE_LABEL,
        mean_bgr=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        augment=False,
    )
    print(dataset)

    # CRF post-processor
    postprocessor = DenseCRF(
        iter_max=CONFIG.CRF.ITER_MAX,
        pos_xy_std=CONFIG.CRF.POS_XY_STD,
        pos_w=CONFIG.CRF.POS_W,
        bi_xy_std=CONFIG.CRF.BI_XY_STD,
        bi_rgb_std=CONFIG.CRF.BI_RGB_STD,
        bi_w=CONFIG.CRF.BI_W,
    )

    # Path to logit files
    logit_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "features",
        "best_checkpoint",
        CONFIG.EXP.ID,
        CONFIG.MODEL.NAME.lower(),
        CONFIG.DATASET.SPLIT.VAL,
        "logit",
    )
    print("Logit src:", logit_dir)

    vis_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "visualization_crf",
        "best_checkpoint",
        CONFIG.EXP.ID,
        CONFIG.DATASET.SPLIT.VAL,
        "ALL"
    )
    makedirs(vis_dir)
    print("Visualization dst:", vis_dir)

    if not os.path.isdir(logit_dir):
        print("Logit not found, run first: python main.py test [OPTIONS]")
        quit()

    # Path to save scores
    save_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "scores",
        "best_checkpoint",
        CONFIG.EXP.ID,
        f"{margin}_{ewc_reg}_{ewc_iter}",
        CONFIG.MODEL.NAME.lower(),
        CONFIG.DATASET.SPLIT.VAL
    )
    makedirs(save_dir)
    save_path = os.path.join(save_dir, "scores_crf.json")
    print("Score dst:", save_path)
    save_path_IAmIoU = os.path.join(save_dir, "scores_IAmIoU_crf.json")
    print("Score dst:", save_path)

    iu_stack = np.zeros(CONFIG.DATASET.N_CLASSES)
    class_stack = np.zeros(CONFIG.DATASET.N_CLASSES)
    iu_small_stack = np.zeros(CONFIG.DATASET.N_CLASSES)
    class_small_stack = np.zeros(CONFIG.DATASET.N_CLASSES)
    iu_medium_stack = np.zeros(CONFIG.DATASET.N_CLASSES)
    class_medium_stack = np.zeros(CONFIG.DATASET.N_CLASSES)
    iu_large_stack = np.zeros(CONFIG.DATASET.N_CLASSES)
    class_large_stack = np.zeros(CONFIG.DATASET.N_CLASSES)
    hist = np.zeros((CONFIG.DATASET.N_CLASSES, CONFIG.DATASET.N_CLASSES))

    # Process per sample
    def process(i):
        image_id, image, gt_label, gt_inst = dataset.__getitem__(i)

        filename = os.path.join(logit_dir, image_id + ".npy")
        logit = np.load(filename)

        _, H, W = image.shape
        logit = torch.FloatTensor(logit)[None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()

        image = image.astype(np.uint8).transpose(1, 2, 0)
        prob = postprocessor(image, prob)
        label = np.argmax(prob, axis=0)

        filename = os.path.join(vis_dir, image_id + ".png")
        pred_image = Image.fromarray(label.astype('uint8'))

        def make_colormap(num=256):
            def bit_get(val, idx):
                return (val >> idx) & 1
            
            colormap = np.zeros((num, 3), dtype=int)
            ind = np.arange(num, dtype=int)
            
            for shift in reversed(list(range(8))):
                for channel in range(3):
                    colormap[:, channel] |= bit_get(ind, channel) << shift
                ind >>= 3
            
            return colormap
        
        cmap = make_colormap(256).tolist()
        palette = [value for color in cmap for value in color]
        pred_image.putpalette(palette)
        pred_image.save(filename)

        return label, gt_label, gt_inst

    # CRF in multi-process
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")(
        [joblib.delayed(process)(i) for i in range(len(dataset))]
    )

    preds, gts, ins_gts = zip(*results)

    # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
    result_IAmIoU = IAmIoU(ins_gts, gts, preds, n_class=CONFIG.DATASET.N_CLASSES)
    result_miou = mIoU(gts, preds, n_class=CONFIG.DATASET.N_CLASSES)

    iu_stack += result_IAmIoU["IoU stack"]
    class_stack += result_IAmIoU["Class stack"]
    iu_small_stack += result_IAmIoU["Small IoU stack"]
    class_small_stack += result_IAmIoU["Small Class stack"]
    iu_medium_stack += result_IAmIoU["Medium IoU stack"]
    class_medium_stack += result_IAmIoU["Medium Class stack"]
    iu_large_stack += result_IAmIoU["Large IoU stack"]
    class_large_stack += result_IAmIoU["Large Class stack"]
    hist += result_miou["hist"]

    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(CONFIG.DATASET.N_CLASSES), iu))

    iu_stack /= class_stack
    iu_mean = np.nanmean(iu_stack)
    cls_iu_mean = dict(zip(range(CONFIG.DATASET.N_CLASSES), iu_stack))

    iu_small_stack /= class_small_stack
    small_iu_mean = np.nanmean(iu_small_stack)
    small_cls_iu_mean = dict(zip(range(CONFIG.DATASET.N_CLASSES), iu_small_stack))

    iu_medium_stack /= class_medium_stack
    medium_iu_mean = np.nanmean(iu_medium_stack)
    medium_cls_iu_mean = dict(zip(range(CONFIG.DATASET.N_CLASSES), iu_medium_stack))

    iu_large_stack /= class_large_stack
    large_iu_mean = np.nanmean(iu_large_stack)
    large_cls_iu_mean = dict(zip(range(CONFIG.DATASET.N_CLASSES), iu_large_stack))

    score_IAmIoU = {
         "Mean IoU": iu_mean,
         "Class IoU": cls_iu_mean,
         "Small Mean IoU": small_iu_mean,
         "Small Class IoU": small_cls_iu_mean,
         "Medium Mean IoU": medium_iu_mean,
         "Medium Class IoU": medium_cls_iu_mean,
         "Learge Mean IoU": large_iu_mean,
         "Large Class IoU": large_cls_iu_mean,
     }
    score = {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }
        
    with open(save_path, "w") as f:
        json.dump(score, f, indent=4, sort_keys=True)
    with open(save_path_IAmIoU, "w") as g:
        json.dump(score_IAmIoU, g, indent=4, sort_keys=True)



if __name__ == "__main__":
    main()
