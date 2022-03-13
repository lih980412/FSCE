import copy
import gc

from fsdet.modeling import build_model
from fsdet.solver import build_lr_scheduler, build_optimizer
# from train_net import default_argument_parser, launch, get_cfg, set_global_cfg, default_setup

# def setup(args):
#     """
#     Create configs and perform basic setups.
#     """
#     cfg = get_cfg()                         # 获取默认cfg
#     cfg.merge_from_file(r"D:\UserD\Li\FSCE-1\configs\MYDATASET_DLA60\base_training.yml")   # 将.yml的参数覆盖到默认cfg中
#
#     cfg.MODEL.DLA.ARCH = 'DLA-60'
#
#     cfg.freeze()                            # 使cfg不可变
#     set_global_cfg(cfg)
#     default_setup(cfg, args)                # 输出cfg相关信息
#     return cfg
#
#
# def main(cfg):
#     cfg = setup(cfg)
#
#     model = build_model(cfg)
#     optimizer = build_optimizer(cfg, model)
#     # data_loader = build_train_loader(cfg)
#
#
# if __name__ == "__main__":
#     args = default_argument_parser().parse_args()
#     print("Command Line Args:", args)
#     launch(
#         main,
#         args.num_gpus,
#         num_machines=args.num_machines,
#         machine_rank=args.machine_rank,
#         dist_url=args.dist_url,
#         args=(args,),
#     )

'2'
import sys
# sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')
import os
import math
import time
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter
import scipy as sp
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
from functools import partial
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from albumentations import (Compose, Normalize, Resize, RandomResizedCrop, HorizontalFlip, VerticalFlip,
                            ShiftScaleRotate, Transpose)
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
import timm
import warnings
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')
from matplotlib import pyplot as plt
import joblib










def get_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')


def init_logger(OUTPUT_DIR):
    log_dir = os.path.join(OUTPUT_DIR, "log", "log_")
    log_file = log_dir + "train.log"
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_transforms(*, data):
    if data == 'train':
        return Compose([
            # RandomResizedCrop(CFG.size, CFG.size),
            Resize(CFG.size, CFG.size),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            Resize(CFG.size, CFG.size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (eta %s)' % (asMinutes(s), asMinutes(rs))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# class ArcMarginProduct(nn.Module):
#     def __init__(self, in_features, out_features):
#         super().__init__()
#         self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.weight)
#         # stdv = 1. / math.sqrt(self.weight.size(1))
#         # self.weight.data.uniform_(-stdv, stdv)
#
#     def forward(self, features):
#         cosine = F.linear(F.normalize(features), F.normalize(self.weight))
#         return cosine

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0,
                 m=0.30, easy_margin=False, ls_eps=0.0, device="cpu"):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.device = device

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight)) # F.normalize(input)、F.normalize(self.weight) 是公式中对输入和权重的正则化
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m   # 这里是三角公式
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        # one_hot = torch.zeros(cosine.size(), device=self.device)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + \
               '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
               ', ' + 'eps=' + str(self.eps) + ')'




'model'
class CustomResNext(nn.Module):
    def __init__(self, model_name='resnext50_32x4d', pretrained=False, device="cpu"):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.model.classifier.in_features
        # self.model.fc = nn.Linear(in_features, CFG.target_size)
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        # self.embedding = nn.Linear(in_features, embedding_size)
        # self.embedding = nn.Sequential(nn.Linear(in_features, CFG.noNeurons),
        #                                nn.BatchNorm1d(CFG.noNeurons),
        #                                nn.PReLU())
                                       # nn.Dropout(p=0.2),
                                       # nn.Linear(CFG.noNeurons, CFG.embedding_size),
                                       # nn.BatchNorm1d(CFG.embedding_size),
                                       # nn.PReLU(num_parameters=1, init=0.25),
                                       # nn.Dropout(p=0.2))
        self.embedding = nn.Sequential(nn.Linear(in_features, CFG.embedding_size),
                                       nn.BatchNorm1d(CFG.embedding_size),
                                       nn.PReLU())
                                       # nn.Dropout(p=0.2),
                                       # nn.Linear(CFG.noNeurons, CFG.embedding_size),
                                       # nn.BatchNorm1d(CFG.embedding_size),
                                       # nn.PReLU(num_parameters=1, init=0.25),
                                       # nn.Dropout(p=0.2))
        self.arcface = ArcMarginProduct(CFG.embedding_size,
                                   CFG.target_size,
                                   s=CFG.s,
                                   m=CFG.m,
                                   easy_margin=CFG.easy_margin,
                                   ls_eps=CFG.ls_eps,
                                   device=device)


    def forward(self, img, target):
        x = self.model(img)
        # return x
        gem_pool = self.pooling(x).flatten(1)
        embedding = self.embedding(gem_pool)
        pred = self.arcface(embedding, target)
        return pred, embedding


'dataset'
class TrainDataset(Dataset):
    def __init__(self, df, transform=None, data_path=None):
        self.df = df
        self.file_names = df['image'].values
        self.labels = df['individual_id'].values
        # self.labels = df['species'].values
        if 'box' in df.columns:
            self.box = [df['box'].values[i]  for i in range(len(df['box'].values))]
        self.transform = transform
        self.data_path = data_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{self.data_path}/{file_name}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        label = torch.tensor(self.labels[idx]).long()
        return image, label





def train_one_epoch(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    model.train()
    start = end = time.time()
    global_step = 0
    for step, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        y_preds, _ = model(images, labels)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        else:
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            LOGGER.info('Epoch: [{0}][{1}/{2}] '
                  'Data_Time {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Process_Time {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'lr: {lr} '
                  'Grad: {grad_norm:.4f}  '
                  .format(
                   epoch+1, step, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   remain=timeSince(start, float(step+1)/len(train_loader)),
                   grad_norm=grad_norm,
                   lr=optimizer.param_groups[0]["lr"]
                   ))

    return losses.avg

@torch.inference_mode()
def valid_fn(valid_loader, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (images, labels) in enumerate(valid_loader):
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds, _ = model(images, labels)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.softmax(dim=1).to('cpu').numpy())
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            LOGGER.info('EVAL: [{0}/{1}] '
                  'Data_Time {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Process_Time {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(
                   step, len(valid_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   remain=timeSince(start, float(step+1)/len(valid_loader)),
                   ))
    predictions = np.concatenate(preds)
    return losses.avg, predictions

def train_net(folds, fold, train_path, test_path, device):

    LOGGER.info(f"========== fold: {fold} training ==========")

    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)

    train_dataset = TrainDataset(train_folds, transform=get_transforms(data='train'), data_path=train_path)
    valid_dataset = TrainDataset(valid_folds, transform=get_transforms(data='valid'), data_path=test_path)

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size,
                              shuffle=True, num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size_val,
                              shuffle=False, num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    model = CustomResNext(CFG.model_name, pretrained=True, device=device)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
    # optimizer = SGD(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, verbose=True,
    #                               eps=CFG.eps)
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    criterion = nn.CrossEntropyLoss()
    # criterion = softmax_cross_entropy_loss_label_smooth(CFG.smoothing)

    if CFG.resume:
        if os.path.isfile(CFG.resume_file):
            checkpoint = torch.load(CFG.resume_file)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found")
    else:
        start_epoch = 0

    best_val_loss = np.inf
    for epoch in range(start_epoch, CFG.epochs):
        gc.collect()
        start_time = time.time()
        # avg_loss = 0
        avg_loss = train_one_epoch(train_loader, model, criterion, optimizer, epoch, scheduler, device)
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)
        scheduler.step()
        # valid_labels = valid_folds[CFG.target_col].values
        # score = map_per_set(valid_labels, preds)
        elapsed = time.time() - start_time
        LOGGER.info(
            f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        # LOGGER.info(f'Epoch {epoch + 1} - Accuracy: {score}')
        if avg_val_loss <= best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            LOGGER.info(f'Epoch {epoch + 1} - Save Best Loss: {best_val_loss:.4f} Model')
            torch.save({'model': model.state_dict(), 'preds': preds},
                       OUTPUT_DIR + f'{CFG.model_name}_fold{fold}_best.pth')

    model.load_state_dict(best_model_wts)

    # check_point = torch.load(OUTPUT_DIR + f'{CFG.model_name}_fold{fold}_best.pth')
    # valid_folds[[str(c) for c in range(5)]] = check_point['preds']
    # valid_folds['preds'] = check_point['preds']

    return model




class softmax_cross_entropy_loss_label_smooth(nn.Module):
    def __init__(self, smoothing):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1 - smoothing

    def forward(self, pred_class_logits, gt_classes):
        logprobs = F.log_softmax(pred_class_logits, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=gt_classes.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# def get_result(result_df):
#     preds = result_df['preds'].values
#     labels = result_df[CFG.target_col].values
#     score = map_per_set(labels, preds)
#     LOGGER.info(f'Score: {score:<.5f}')


'cfg'
class CFG:
    print_freq = 100
    num_workers = 1
    # model_name = 'resnext50_32x4d'
    # model_name = 'efficientnet_b0'
    model_name = 'tf_efficientnet_b4'
    size = 448
    # size = 224
    epochs = 15
    factor = 0.2
    patience = 5
    eps = 1e-6
    T_max = 5
    lr = 1e-4
    min_lr = 1e-6
    batch_size = 8
    batch_size_val = 32
    weight_decay = 1e-6
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = 42
    target_size = 15587
    target_col = 'individual_id'
    n_fold = 3
    trn_fold = [0, 1, 2, 3, 4]
    noNeurons = 256
    embedding_size = 512
    # ArcFace Hyperparameters
    s = 30.0
    m = 0.50
    ls_eps = 0.0
    easy_margin = False
    MARGIN_MAX = 0.4
    MARGIN_MIN = 0.2
    # Label Smoothing
    smoothing = 0.1

    resume = True
    resume_file = r""


# 'MAP@5, https://www.kaggle.com/pestipeti/explanation-of-map5-scoring-metric'
def map_per_image(label, predictions):
    """Computes the precision score of one image.

    Parameters
    ----------
    label : string
            The true label of the image
    predictions : list
            A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """
    try:
        # return 1 / (predictions[:5].index(label) + 1)
        indice = torch.where(torch.topk(torch.tensor(predictions), k=5, dim=0, largest=True, sorted=True)[1] == label)[0]
        if len(indice) == 0:
            return 0.0
        return 1 / (indice.cpu().numpy() + 1)
    except ValueError:
        return 0.0

def map_per_set(labels, predictions):
    """Computes the average over multiple images.

    Parameters
    ----------
    labels : list
             A list of the true labels. (Only one true label per images allowed!)
    predictions : list of list
             A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """
    return np.mean([map_per_image(l, p) for l, p in zip(labels, predictions)])



if __name__ == "__main__":
    OUTPUT_DIR = 'F:\Dataset\Kaggle\output'
    TRAIN_PATH = r'F:\Dataset\Kaggle\train_images'
    TEST_PATH = r'F:\Dataset\Kaggle\test_images'
    ROOT_PATH = r"F:\Dataset\Kaggle"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_torch(seed=CFG.seed)
    LOGGER = init_logger(OUTPUT_DIR)

    # Load training data
    '三种train.csv'
    '1'
    # train_file = pd.read_csv(r'F:\Dataset\Kaggle\train.csv')
    # # train_file = pd.read_csv(r'F:\Dataset\Kaggle\train2.csv')
    # # PreProcess
    # train_file['species'].replace({
    #     'bottlenose_dolpin': 'bottlenose_dolphin',
    #     'kiler_whale': 'killer_whale',
    #     'beluga': 'beluga_whale',
    #     'globis': 'short_finned_pilot_whale',
    #     'pilot_whale': 'short_finned_pilot_whale'
    # }, inplace=True)
    #
    # label2name = {}
    # for i in range(len(train_file['individual_id'].unique())):
    #     label2name[i] = train_file['individual_id'].unique()[i]
    # name2label = dict((v, k) for k, v in label2name.items())
    # train_file['individual_id'].replace(name2label, inplace=True)
    '2'
    train_file = pd.read_csv(r'F:\Dataset\Kaggle\train1.csv')
    folds = train_file.copy()
    # folds = train_file.copy()
    Fold = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[CFG.target_col])):
        folds.loc[val_index, 'fold'] = int(n)
    folds['fold'] = folds['fold'].astype(int)
    # Split into folds for cross validation - we used the same split for all the models we trained!
    # folds = train.merge(
    #     pd.read_csv("../input/cassava-leaf-disease-resnext/validation_data.csv")[["image_id", "fold"]], on="image_id")
    '3'
    # data = pd.read_csv(os.path.join(ROOT_PATH, "train.csv"))
    #
    # data['species'].replace({
    #     'bottlenose_dolpin': 'bottlenose_dolphin',
    #     'kiler_whale': 'killer_whale',
    #     'beluga': 'beluga_whale',
    #     'globis': 'short_finned_pilot_whale',
    #     'pilot_whale': 'short_finned_pilot_whale'
    # }, inplace=True)
    # N_CLASSES = len(data["individual_id"].unique())
    # # Computing an integer mapping for individuals ids
    # individual_mapping = {k: i for i, k in enumerate(data["individual_id"].unique())}
    # # Compute margins for ArcFaces with dynamic margins
    # tmp = np.sqrt(1 / np.sqrt(data['individual_id'].value_counts().loc[list(individual_mapping)].values))
    # MARGINS = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * (CFG.MARGIN_MAX - CFG.MARGIN_MIN) + CFG.MARGIN_MIN
    #
    # data["individual_id_integer"] = data["individual_id"].map(individual_mapping)
    # folds = data.copy()
    # # folds = train_file.copy()
    # Fold = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    # for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[CFG.target_col])):
    #     folds.loc[val_index, 'fold'] = int(n)
    # folds['fold'] = folds['fold'].astype(int)
    # folds.to_csv(r"F:\Dataset\Kaggle\fold.csv")
    '4'
    # train_file = pd.read_csv(r'F:\Dataset\Kaggle\fold.csv')
    # print(folds.groupby(['fold', CFG.target_col]).size())

    oof_df = pd.DataFrame()
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            # _oof_df = train_net(folds, fold, TRAIN_PATH, TRAIN_PATH, device)
            # oof_df = pd.concat([oof_df, _oof_df])
            # LOGGER.info(f"========== fold: {fold} result ==========")
            #
            # get_result(_oof_df)

            model = train_net(folds, fold, TRAIN_PATH, TRAIN_PATH, device)


    LOGGER.info(f"========== CV ==========")
    get_result(oof_df)
    oof_df.to_csv(OUTPUT_DIR + 'oof_df.csv', index=False)
