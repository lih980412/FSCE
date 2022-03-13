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
from tqdm import tqdm

from kaggle.train1_ import GeM, Swish_module, ArcMarginProduct_subcenter, ArcFaceLossAdaptiveMargin


class WandDIDPred(Dataset):
    def __init__(self, data, folder, base_path):
        self.base_path = os.path.join(base_path, folder)
        self.data = data
        self.labels = 0
        # Augmentations
        transformations = albumentations.Compose([
            albumentations.Resize(CFG.IMAGE_SIZE, CFG.IMAGE_SIZE),
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
        image = self.preprocess(self.data["image"].iloc[idx])
        label = self.labels
        return image, label

    def preprocess(self, image):
        image = os.path.join(self.base_path, image)
        image = cv2.imread(image)[:, :, ::-1]
        if image.shape[0] < CFG.IMAGE_SIZE or image.shape[1] < CFG.IMAGE_SIZE:
            image = cv2.resize(image, (CFG.IMAGE_SIZE, CFG.IMAGE_SIZE), cv2.INTER_CUBIC)
        if self.transforms is not None:
            image = self.transforms(image)["image"]
        return image

    def __len__(self):
        return len(self.data)


class Solution:
    def __init__(self, database, q_prior, individual_mapping):
        self.database_embeddings = np.array(database["embeddings"]["embedding"].values.tolist())
        self.database_individuals = database["embeddings"]["individual_id"].values
        self.q_prior = q_prior
        self.embed_neigh = NearestNeighbors(n_neighbors=CFG.KNN, metric='cosine')
        self.embed_neigh.fit(self.database_embeddings)
        self.class_neigh = NearestNeighbors(n_neighbors=CFG.KNN, metric='cosine')
        self.class_neigh.fit(database["class_centers"])
        self.default = ['938b7e931166', '5bf17305f073', '7593d2aee842', '7362d7a01d00', '956562ff2888']
        self.individual_mapping = individual_mapping

    def predict(self, queries):
        embed_distances, embed_idxs = self.embed_neigh.kneighbors(queries, CFG.KNN, return_distance=True)
        class_distances, class_idxs = self.class_neigh.kneighbors(queries, CFG.KNN, return_distance=True)

        class_individuals = np.repeat(list(self.individual_mapping), CFG.CENTERS_PER_CLASS)[class_idxs]
        embed_individuals = self.database_individuals[embed_idxs]

        n = embed_distances.size
        embeddings_df = pd.DataFrame(data={
            'distance': embed_distances.ravel(),
            'individual': embed_individuals.ravel(),
            'query_id': np.repeat(np.arange(len(queries)), CFG.KNN)
        }, index=np.arange(n))

        class_df = pd.DataFrame(data={
            'distance': class_distances.ravel(),
            'individual': class_individuals.ravel(),
            'query_id': np.repeat(np.arange(len(queries)), CFG.KNN)
        }, index=np.arange(n))

        embeddings_topk = embeddings_df.groupby(["query_id", "individual"]).agg("min")['distance'].groupby('query_id', group_keys=False).nsmallest(5)
        class_topk = class_df.groupby(["query_id", "individual"]).agg("min")['distance'].groupby('query_id', group_keys=False).nsmallest(5)
        embeddings_topk = embeddings_topk.reset_index().groupby("query_id").agg(list)
        class_topk = class_topk.reset_index().groupby("query_id").agg(list)
        class_t_new = np.quantile(class_topk["distance"].apply(lambda x: x[0]), 1 - self.q_prior)
        embeddings_t_new = np.quantile(embeddings_topk["distance"].apply(lambda x: x[0]), 1 - self.q_prior)

        def insert_new_individuals(x):
            m = np.array(x["distance"]) > class_t_new
            preds = x["individual"]
            if m.any():
                preds.insert(np.argmax(m), "new_individual")
            preds = preds + [y for y in self.default if y not in preds]
            return preds[:5]

        preds = class_topk.apply(insert_new_individuals, axis=1)
        return preds.values.tolist()

@torch.inference_mode()
def inference(model, dataloader, device):

    model.eval()
    outputList=[]
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data[0].to(device, dtype=torch.float)
        labels = data[1].to(device, dtype=torch.long)
        _, outputs = model(images, labels)
        outputList.extend(outputs)
    return outputList


class WandDIDNet(LightningModule):
    def __init__(self, continuous_scheduler=True, s=30, m=0.3, margin=None, dataloader=None, out_dim=None, centerperclass=None):
        super().__init__()
        self.save_hyperparameters()
        self.continuous_scheduler = continuous_scheduler
        # Layers
        self.feature_extractor = timm.create_model(CFG.MODEL_NAME, pretrained=True)
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
        return logits, F.normalize(emb)


class CFG:
    SEED = 69
    ### Dataset
    ## Effective batch size will be BATCH_SIZE*ACCUMULATE_GRAD_BATCHES
    #BATCH_SIZE = 16
    BATCH_SIZE = 96
    #ACCUMULATE_GRAD_BATCHES = 8
    ACCUMULATE_GRAD_BATCHES = 1
    IMAGE_SIZE = 512
    #IMAGE_SIZE = 224
    NUM_WORKERS = 0
    ### Model
    MODEL_NAME = "efficientnet_b0"
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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BASE_PATH = r"F:\Dataset\Kaggle"
    data = pd.read_csv(os.path.join(BASE_PATH, "fold.csv"))
    weights_path = r"F:\Dataset\Kaggle\output\efficientnet_b0_fold0_best.pth"

    N_CLASSES = len(data["individual_id"].unique())
    individual_mapping = {k: i for i, k in enumerate(data["individual_id"].unique())}
    tmp = np.sqrt(1 / np.sqrt(data['individual_id'].value_counts().loc[list(individual_mapping)].values))
    MARGINS = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * (CFG.MARGIN_MAX - CFG.MARGIN_MIN) + CFG.MARGIN_MIN


    pred_loader = torch.utils.data.DataLoader(
        WandDIDPred(data, "train_images", base_path=BASE_PATH),
        batch_size=CFG.BATCH_SIZE,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
        shuffle=False
    )

    model = WandDIDNet(margin=MARGINS, out_dim=N_CLASSES, centerperclass=CFG.CENTERS_PER_CLASS)
    model.to(device)
    model.load_state_dict(torch.load(weights_path)['model'])
    preds = inference(model, pred_loader, device)
    preds = torch.stack(preds, dim=0)
    train_data = data.copy()
    train_data["embedding"] = preds.tolist()
    # train_data.to_csv("train.csv")

    # Prediction on test data
    test_data = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"), index_col="image")
    if "inference_image" in test_data.columns:
        test_data["image"] = test_data["inference_image"]
    else:
        test_data["image"] = test_data.index

    test_loader = torch.utils.data.DataLoader(
        WandDIDPred(test_data, "test_images", base_path=BASE_PATH),
        batch_size=CFG.BATCH_SIZE,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
        shuffle=False
    )
    preds = inference(model, test_loader, device)
    preds = torch.stack(preds, dim=0)
    test_data["embedding"] = preds.tolist()


    train_data["individual_id_integer"] = train_data["individual_id"].map(individual_mapping).fillna(-1)
    train_embeddings = np.array(train_data["embedding"].values.tolist())
    test_embeddings = np.array(test_data["embedding"].values.tolist())
    class_centers = model.metric_classify.weight.detach().cpu().numpy()

    solution = Solution({
        "embeddings": train_data,
        "class_centers": class_centers
    }, CFG.Q_NEW, individual_mapping)
    predictions = solution.predict(test_embeddings)
    predictions = pd.Series(predictions, test_data.index, name="predictions").map(lambda x: " ".join(x))
    predictions.to_csv(r"F:\Dataset\Kaggle\submission1.csv")

if __name__ == "__main__":
    main()