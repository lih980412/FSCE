import gc
import time

import numpy as np
import pandas as pd
import os
import torch
import random
import albumentations
import cv2
import math
import copy
import torchmetrics
import torchvision
import pytorch_lightning as pl
# import wandb
import json
from torch import nn, Tensor
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms, models
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from sklearn.neighbors import NearestNeighbors
import timm
from torch.cuda import amp

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
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (eta %s)' % (asMinutes(s), asMinutes(rs))

# def trivial_batch_collator(batch):
#     """
#     A batch collator that does nothing.
#     """
#     return batch
# def my_collate(batch):
#     data = torch.stack([item[0] for item in batch])
#     data = [item[0] for item in batch]
#     target = [item[1] for item in batch]
#     target = torch.LongTensor(target)
#     return [data, target]

class WandDID(Dataset):
    def __init__(self, data, augment=True, base_path=None):
        self.base_path = os.path.join(base_path, "train_images")
        self.data = data
        # Augmentations
        if augment:
            transformations = albumentations.Compose([
                albumentations.augmentations.transforms.HueSaturationValue(p=0.5),
                # albumentations.CoarseDropout(max_holes=1, max_height=(CFG.IMAGE_SIZE // 9), max_width=(CFG.IMAGE_SIZE // 9),
                #                       p=0.5),
                albumentations.Resize(CFG.IMAGE_SIZE, CFG.IMAGE_SIZE),
                albumentations.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=(-0.2, 0.2), p=0.5),
                albumentations.Normalize(),
                ToTensorV2(p=1.0)
            ])
        else:
            transformations = albumentations.Compose([
                albumentations.Normalize(),
                ToTensorV2(p=1.0)
            ])

        def make_transform(transform=False):
            def f(image):
                if transform:
                    image_np = np.array(image)
                    augmented = transform(image=image_np)
                return augmented

            return f

        self.transforms = transforms.Compose([
            transforms.Lambda(make_transform(transformations)),
        ])

    def __getitem__(self, idx):
        image = self.preprocess(self.data["image"].iloc[idx], idx)

        label = self.data["individual_id_integer"].iloc[idx]
        return image, torch.tensor(label, dtype=torch.long)

    def preprocess(self, image, idx):
        image = os.path.join(self.base_path, image)
        # if not np.isnan(self.data["bbox"].iloc[idx]):   # 非空
        #     bbox = self.data["bbox"].iloc[idx]


        image = cv2.imread(image)[:, :, ::-1]
        if type(self.data["bbox"].iloc[idx]) is not float:
            bbox = (self.data["bbox"].iloc[idx]).split(" ")
            image = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :]
        if image.shape[0] < CFG.IMAGE_SIZE or image.shape[1] < CFG.IMAGE_SIZE:
            image = cv2.resize(image, (CFG.IMAGE_SIZE, CFG.IMAGE_SIZE), cv2.INTER_CUBIC)
        if self.transforms is not None:
            image = self.transforms(image)["image"]
        return image

    def plot_sample(self, idx):
        image = self.data["image"].iloc[idx]
        image = os.path.join(self.base_path, image)
        image = cv2.imread(image)[:, :, ::-1]
        plt.title("{} ({})".format(
            self.data["individual_id"].iloc[idx],
            self.data["species"].iloc[idx]
        ))
        plt.imshow(image)
        plt.show()

    def __len__(self):
        return len(self.data)


class PeakScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self, optimizer,
            epoch_size=-1,
            lr_start=0.000001,
            lr_max=0.000005,
            lr_min=0.000001,
            lr_ramp_ep=5,
            lr_sus_ep=0,
            lr_decay=0.8,
            verbose=True
    ):
        self.epoch_size = epoch_size
        self.optimizer = optimizer
        self.lr_start = lr_start
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_ramp_ep = lr_ramp_ep
        self.lr_sus_ep = lr_sus_ep
        self.lr_decay = lr_decay
        self.is_plotting = True
        epochs = list(range(CFG.EPOCHS))
        learning_rates = []
        for i in epochs:
            self.epoch = i
            learning_rates.append(self.get_lr())
        self.is_plotting = False
        self.epoch = 0
        plt.scatter(epochs, learning_rates)
        plt.show()
        super(PeakScheduler, self).__init__(optimizer, verbose=verbose)

    def get_lr(self):
        if not self.is_plotting:
            if self.epoch_size == -1:
                self.epoch = self._step_count - 1
            else:
                self.epoch = (self._step_count - 1) / self.epoch_size
        print(self.epoch)
        if self.epoch < self.lr_ramp_ep:
            lr = (self.lr_max - self.lr_start) / self.lr_ramp_ep * self.epoch + self.lr_start

        elif self.epoch < self.lr_ramp_ep + self.lr_sus_ep:
            lr = self.lr_max
        else:
            lr = (self.lr_max - self.lr_min) * self.lr_decay ** (
                        self.epoch - self.lr_ramp_ep - self.lr_sus_ep) + self.lr_min
        return [lr for _ in self.optimizer.param_groups]


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class Swish(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)


class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features * k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine



class ArcFaceLossAdaptiveMargin(nn.modules.Module):
    def __init__(self, margins, out_dim, s):
        super().__init__()
        self.crit = DenseCrossEntropy()
        self.s = s
        self.register_buffer('margins', torch.tensor(margins, device="cuda:0"))
        self.out_dim = out_dim

    def forward(self, logits, labels):
        # ms = []
        # ms = self.margins[labels.cpu().numpy()]
        ms = self.margins[labels]
        cos_m = torch.cos(ms)  # torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.sin(ms)  # torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.cos(math.pi - ms)  # torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.sin(math.pi - ms) * ms  # torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        labels = F.one_hot(labels, self.out_dim)
        labels = labels.half() if CFG.MIXED_PRECISION else labels.float()
        cosine = logits
        sine = torch.sqrt(1.0 - cosine * cosine)
        phi = cosine * cos_m.view(-1, 1) - sine * sin_m.view(-1, 1)
        phi = torch.where(cosine > th.view(-1, 1), phi, cosine - mm.view(-1, 1))
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        return loss


# Make a deterministic pipeline
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #gpu randomseed fixed
    torch.backends.cudnn.deterministic = True


# class HappyWhaleModel(nn.Module):
#     def __init__(self, modelName, numClasses, noNeurons, embeddingSize):
#         super(HappyWhaleModel, self).__init__()
#         self.model = timm.create_model(modelName, pretrained=False)
#         self.embsize = embeddingSize
#         in_features = self.model.classifier.in_features
#         self.model.classifier = nn.Identity()
#         # self.model.global_pool = nn.Identity()
#         self.pooling = GeM()
#         self.drop = nn.Dropout(p=0.2, inplace=False)
#
#         self.fc = nn.Sequential(
#             nn.Linear(in_features, embeddingSize),
#             # nn.AdaptiveAvgPool1d(embeddingSize)
#         )
#
#         # self.avgpool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
#         self.arc_head = ArcMarginProduct(embeddingSize, numClasses)
#         self.class_head = nn.Sequential(
#             # nn.Linear(2048, 512),
#             nn.Linear(embeddingSize, num_classes)
#         )
#
#     def forward(self, images, labels=None):
#         features = self.model.forward_features(images)
#         # pooled_features = self.pooling(features).flatten(1)
#         # pooled_drop = self.drop(pooled_features)
#         features = self.pooling(features)
#         #emb = self.fc(features.flatten(1))
#         emb = features.flatten(1)
#         emb =self.fc(emb)
#         #emb = F.adaptive_avg_pool1d(emb, 2048)
#         if labels != None:
#
#             arc_output = self.arc_head(emb, labels)
#             class_output = self.class_head(emb)
#             return class_output, arc_output, emb
#         else:
#
#             return emb  # feartures



class WandDIDNet(LightningModule):
    def __init__(self, continuous_scheduler=True, s=30, m=0.3, margin=None, dataloader=None, out_dim=None, centerperclass=None):
        super().__init__()
        self.save_hyperparameters()
        self.continuous_scheduler = continuous_scheduler
        # Layers
        self.feature_extractor = timm.create_model(CFG.model_name, pretrained=True)
        in_features = self.feature_extractor.classifier.in_features
        self.feature_extractor.classifier = nn.Identity()
        self.feature_extractor.global_pool = nn.Identity()
        self.pooling = GeM()
        self.dropout = nn.Dropout()
        self.dense = nn.Linear(in_features, CFG.EMBEDDING_SIZE)
        self.swish = Swish_module()
        self.metric_classify = ArcMarginProduct_subcenter(CFG.EMBEDDING_SIZE, out_dim, centerperclass)
        # Loss
        self.criterion = ArcFaceLossAdaptiveMargin(margins=margin, out_dim=out_dim, s=s)
        # Metrics
        self.train_acc = torchmetrics.Accuracy()
        self.train_top_k_acc = torchmetrics.Accuracy(top_k=5)
        self.val_acc = torchmetrics.Accuracy()
        self.val_top_k_acc = torchmetrics.Accuracy(top_k=5)

        self.train_dataloader = dataloader

    def forward(self, image, label=None):
        """
            Return embedding of the images
        """
        features = self.feature_extractor(image)
        x = self.pooling(features).flatten(1)
        x = self.dropout(x)
        emb = self.dense(x)
        logits = self.metric_classify(emb)
        return logits, emb

    # def training_step(self, batch, batch_idx):
    #     """
    #         Return the loss to do a step on
    #     """
    #     img, label = batch
    #     embedding = self(img)
    #     logits = self.metric_classify(embedding)
    #     loss = self.criterion(logits, label)
    #     # Log metrics
    #     self.train_acc(logits, label)
    #     self.train_top_k_acc(logits, label)
    #     self.log("train/loss", loss)
    #     self.log("train/acc", self.train_acc)
    #     self.log("train/top_k_acc", self.train_top_k_acc)
    #     # Return loss, labels and preds
    #     return {"loss": loss, "preds": logits.detach(), "targets": label.detach()}

    # def configure_optimizers(self):
    #     """
    #         Build optimizer(s) and lr scheduler(s)
    #     """
    #     optimizer = torch.optim.AdamW(self.parameters())
    #     if self.continuous_scheduler:
    #         sched = {
    #             "scheduler": PeakScheduler(optimizer, epoch_size=len(self.train_dataloader) // CFG.ACCUMULATE_GRAD_BATCHES,
    #                                        verbose=False, lr_max=0.000005 * CFG.BATCH_SIZE * CFG.ACCUMULATE_GRAD_BATCHES),
    #             "interval": "step",
    #         }
    #     else:
    #         sched = {
    #             "scheduler": PeakScheduler(optimizer),
    #             "interval": "epoch",
    #         }
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": sched
    #     }

    def save_class_weights(self):
        """
            Save the class centers as a tensor
        """
        torch.save(self.metric_classify.weight, 'class_weights.pt')

def valid_one_epoch(val_loader, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # scores = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (images, labels) in enumerate(val_loader):
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds, _ = model(images, labels)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.softmax(dim=1).to('cpu').numpy())
        # if CFG.gradient_accumulation_steps > 1:
        #     loss = loss / CFG.gradient_accumulation_steps
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(val_loader) - 1):
            LOGGER.info('EVAL: [{0}/{1}] '
                        'Data_Time {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Process_Time {remain:s} '
                        'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                .format(
                step, len(val_loader), batch_time=batch_time,
                data_time=data_time, loss=losses,
                remain=timeSince(start, float(step + 1) / len(val_loader)),
            ))
    predictions = np.concatenate(preds)
    return losses.avg, predictions



def train_one_epoch(train_loader, model, criterion, optimizer, epoch, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # scores = AverageMeter()
    model.train()
    start = end = time.time()
    global_step = 0

    # 混合精度训练
    enable_amp = True if 'cuda' in device.type else False
    if enable_amp:
        LOGGER.info("Using enable_amp")
    scaler = amp.GradScaler(enabled = enable_amp)

    for step, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        if enable_amp:
            with amp.autocast(enabled = enable_amp):
                y_preds, _ = model(images, labels)
                loss = criterion(y_preds, labels)
            losses.update(loss.item(), batch_size)
            scaler.scale(loss).backward()
            # scaler.unscale_(optimizer)
            # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10000)
            scaler.step(optimizer)
            scaler.update()

        else:
            y_preds, _ = model(images, labels)
            loss = criterion(y_preds, labels)
            losses.update(loss.item(), batch_size)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10000)
            optimizer.step()
        optimizer.zero_grad()
        global_step += 1

        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            LOGGER.info('Epoch: [{0}][{1}/{2}] '
                        'Data_Time {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Process_Time {remain:s} '
                        'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                        'lr: {lr} '
                        # 'Grad: {grad_norm:.4f}  '
                .format(
                epoch + 1, step, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses,
                remain=timeSince(start, float(step + 1) / len(train_loader)),
                # grad_norm=grad_norm,
                lr=optimizer.param_groups[0]["lr"]
            ))
    return losses.avg

class CFG:
    SEED = 69
    ### Dataset
    ## Effective batch size will be BATCH_SIZE*ACCUMULATE_GRAD_BATCHES
    BATCH_SIZE = 32
    # BATCH_SIZE = 48
    #ACCUMULATE_GRAD_BATCHES = 8
    ACCUMULATE_GRAD_BATCHES = 1
    IMAGE_SIZE = 512
    #IMAGE_SIZE = 224
    NUM_WORKERS = 0
    ### Model
    model_name = "efficientnet_b0"
    EMBEDDING_SIZE = 512
    ### Training
    ## Arcfaces
    CENTERS_PER_CLASS = 3
    S = 30
    MARGIN_MIN = 0.2
    MARGIN_MAX = 0.4
    EPOCHS = 20
    MIXED_PRECISION = False
    MODEL_PATH = r"F:\Dataset\Kaggle\output"
    # Inference
    KNN = 100
    Q_NEW = 0.112 # Proportion of new individuals expected in the dataset

    continuous_scheduler = True
    print_freq = 100
    resume = True
    # resume_file = r"F:\Dataset\Kaggle\output\efficientnet_b0_fold0_best.pth"
    resume_file = r""




def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fix_seed(CFG.SEED)

    BASE_PATH = r"F:\Dataset\Kaggle"

    # data = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
    # N_CLASSES = len(data["individual_id"].unique())
    # # Computing an integer mapping for individuals ids
    # individual_mapping = {k: i for i, k in enumerate(data["individual_id"].unique())}
    # # Compute margins for ArcFaces with dynamic margins
    # tmp = np.sqrt(1 / np.sqrt(data['individual_id'].value_counts().loc[list(individual_mapping)].values))
    # MARGINS = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * (CFG.MARGIN_MAX - CFG.MARGIN_MIN) + CFG.MARGIN_MIN
    # # Save individual mapping
    # with open("individual_mapping.json", "w") as f:
    #     json.dump(individual_mapping, f)
    #
    # data["individual_id_integer"] = data["individual_id"].map(individual_mapping)


    data = pd.read_csv(os.path.join(BASE_PATH, "fold.csv"))
    N_CLASSES = len(data["individual_id"].unique())
    individual_mapping = {k: i for i, k in enumerate(data["individual_id"].unique())}
    tmp = np.sqrt(1 / np.sqrt(data['individual_id'].value_counts().loc[list(individual_mapping)].values))
    MARGINS = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * (CFG.MARGIN_MAX - CFG.MARGIN_MIN) + CFG.MARGIN_MIN

    trn_idx = data[data['fold'] != 0].index
    val_idx = data[data['fold'] == 0].index

    train_folds = data.loc[trn_idx].reset_index(drop=True)
    valid_folds = data.loc[val_idx].reset_index(drop=True)


    train_dataset = WandDID(train_folds, base_path=BASE_PATH)
    # Dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=CFG.BATCH_SIZE,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
        shuffle=True,
    )

    val_dataset = WandDID(valid_folds, base_path=BASE_PATH)
    # Dataloader
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=CFG.BATCH_SIZE * 2,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
        shuffle=False,
    )
    # train_dataset.plot_sample(random.randint(0, len(train_dataset) - 1))
    model = WandDIDNet(margin=MARGINS, dataloader=train_loader, out_dim=N_CLASSES, centerperclass=CFG.CENTERS_PER_CLASS)
    model.to(device)
    # wandb_logger = WandbLogger(project="W&D - identification")
    # Trainer


    optimizer = torch.optim.AdamW(model.parameters())
    # scheduler = PeakScheduler(optimizer,  verbose=False, lr_max=0.003)
    scheduler = PeakScheduler(optimizer, epoch_size=1,
                                       verbose=False, lr_max=0.001)
    criterion = ArcFaceLossAdaptiveMargin(margins=MARGINS, out_dim=N_CLASSES, s=30)
    criterion.to(device)

    if CFG.resume and CFG.resume_file:
        if os.path.isfile(CFG.resume_file):
            checkpoint = torch.load(CFG.resume_file)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            start_epoch = 0
            print("=> no checkpoint found")
    else:
        print("=> there is no checkpoint")
        start_epoch = 0
        
    best_val_loss = np.inf
    early_stop = 0
    for epoch in range(start_epoch, CFG.EPOCHS):
        gc.collect()
        start_time = time.time()
        # avg_val_loss = 0
        avg_loss = train_one_epoch(train_loader, model, criterion, optimizer, epoch, device)
        avg_val_loss, preds = valid_one_epoch(val_loader, model, criterion, device)
        scheduler.step()
        # valid_labels = valid_folds[CFG.target_col].values
        # score = map_per_set(valid_labels, preds)
        elapsed = time.time() - start_time
        LOGGER.info(
            f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        # LOGGER.info(f'Epoch {epoch + 1} - Accuracy: {score}')
        if avg_val_loss <= best_val_loss:
            early_stop = 0
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            LOGGER.info(f'Epoch {epoch + 1} - Save Best Loss: {best_val_loss:.4f} Model')
            torch.save({'model': model.state_dict(), 'preds': preds, 'epoch':epoch+1, "optimizer": optimizer}, 
                       OUTPUT_DIR + f'\\{CFG.model_name}_fold0_best.pth')
        else:
            early_stop += 1

        if early_stop >= 5:
            print("early stop bacause val loss get stuck")
            break

    model.load_state_dict(best_model_wts)


    # trainer = Trainer(
    #     profiler="simple",  # Profiling
    #     accumulate_grad_batches=CFG.ACCUMULATE_GRAD_BATCHES,  # Accumulate gradient over multiple batches
    #     gpus=1,  # Use the one GPU we have
    #     precision=16 if CFG.MIXED_PRECISION else 32,  # Mixed precision
    #     max_epochs=CFG.EPOCHS,
    #     # logger=wandb_logger,
    #     log_every_n_steps=10
    # )
    # # Let's go ⚡
    # trainer.fit(model, train_loader)

    # trainer.save_checkpoint(CFG.MODEL_PATH)


if __name__ == "__main__":
    OUTPUT_DIR = 'F:\Dataset\Kaggle\output'
    LOGGER = init_logger(OUTPUT_DIR)
    main()

