import pickle

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

# from kaggle.train1_ import GeM, Swish_module, ArcMarginProduct_subcenter, ArcFaceLossAdaptiveMargin
from kaggle.train1_ import GeM, ArcMarginProduct_subcenter, ArcFaceLossAdaptiveMargin





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
        return 1 / (predictions[:5].index(label) + 1)
        # indice = torch.where(torch.topk(torch.tensor(predictions), k=5, dim=0, largest=True, sorted=True)[1] == label)[0]
        # if len(indice) == 0:
        #     return 0.0
        # return 1 / (indice.cpu().numpy() + 1)
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

import faiss
def PredictGrid(train_cnn_predictions, valid_cnn_predictions, train_labels, valid_labels, new_individual_thres, class_centers=None, individual_mapping=None):
    'faiss'
    # index = faiss.IndexFlatL2(512)
    # index.add(train_cnn_predictions.astype(np.float32))
    # distances, idxs = index.search(valid_cnn_predictions.astype(np.float32), 100)

    'ml'
    neigh = NearestNeighbors(n_neighbors=CFG.KNN, metric="cosine")
    neigh.fit(train_cnn_predictions)
    # class_neigh = NearestNeighbors(n_neighbors=CFG.KNN, metric='cosine')
    # class_neigh.fit(class_centers)

    distances, idxs = neigh.kneighbors(valid_cnn_predictions, return_distance=True)
    # cls_distances, idx = class_neigh.kneighbors(valid_cnn_predictions, return_distance=True)

    conf = 1 - distances
    # cls_conf = 1 - cls_distances

    # n = distances.size
    # class_individuals = np.repeat(list(individual_mapping), CFG.CENTERS_PER_CLASS)[idxs]
    # class_df = pd.DataFrame(data={
    #     'distance': distances.ravel(),
    #     'individual': class_individuals.ravel(),
    #     'query_id': np.repeat(np.arange(len(valid_cnn_predictions)), CFG.KNN)
    # }, index=np.arange(n))
    # class_topk = class_df.groupby(["query_id", "individual"]).agg("min")['distance'].groupby('query_id', group_keys=False).nsmallest(5)
    # class_topk = class_topk.reset_index().groupby("query_id").agg(list)
    # class_t_new = np.quantile(class_topk["distance"].apply(lambda x: x[0]), 1 - self.q_prior)


    preds = []

    for j in range(len(idxs)):
        preds.append(list(train_labels[idxs[j]]))

    allTop5Preds = []
    valid_labels_list = []
    for i in range(len(preds)):
        valid_labels_list.append((valid_labels[i]))

        predictTop = preds[i][:5]
        Top5Conf = conf[i][:5]

        if Top5Conf[0] < new_individual_thres:

            tempList = ['new_individual', predictTop[0], predictTop[1], predictTop[2], predictTop[3]]
            allTop5Preds.append(tempList)

        elif Top5Conf[1] < new_individual_thres:

            tempList = [predictTop[0], 'new_individual', predictTop[1], predictTop[2], predictTop[3]]
            allTop5Preds.append(tempList)

        elif Top5Conf[2] < new_individual_thres:

            tempList = [predictTop[0], predictTop[1], 'new_individual', predictTop[2], predictTop[3]]
            allTop5Preds.append(tempList)

        elif Top5Conf[3] < new_individual_thres:

            tempList = [predictTop[0], predictTop[1], predictTop[2], 'new_individual', predictTop[3]]
            allTop5Preds.append(tempList)

        elif Top5Conf[4] < new_individual_thres:

            tempList = [predictTop[0], predictTop[1], predictTop[2], predictTop[3], 'new_individual']
            allTop5Preds.append(tempList)

        else:
            allTop5Preds.append(predictTop)

        if (('new_individual' in allTop5Preds[-1]) and (valid_labels_list[i] not in train_labels)):
            allTop5Preds[-1] = [valid_labels_list[i] if x == 'new_individual' else x for x in allTop5Preds[-1]]

    score = map_per_set(valid_labels_list, allTop5Preds)

    return score


def GetSubmission(train_data, valid_data, train_labels, neighbors=100, metric='cosine', new_individual_thres=0.6, label2name=None):
    neigh = NearestNeighbors(n_neighbors=neighbors, metric=metric)
    neigh.fit(train_data)
    distances, idxs = neigh.kneighbors(valid_data, return_distance=True)

    'faiss'
    # index = faiss.IndexFlatL2(512)
    # index.add(train_data.astype(np.float32))
    # distances, idxs = index.search(valid_data.astype(np.float32), neighbors)

    conf = 1 - distances
    preds = []
    df = pd.read_csv(r'F:\Dataset\Kaggle\sample_submission.csv')
    for i in range(len(idxs)):

        preds.append(train_labels[idxs[i]])
        predictTopDecoded = {}
    for i in range(len(distances)):

        predictTop = list(preds[i][:5])
        topValues = conf[i][:5]

        if topValues[0] < new_individual_thres:

            tempList = ['new_individual', predictTop[0], predictTop[1], predictTop[2], predictTop[3]]
            predictTopDecoded[df.iloc[i]['image']] = tempList

        elif topValues[1] < new_individual_thres:

            tempList = [predictTop[0], 'new_individual', predictTop[1], predictTop[2], predictTop[3]]
            predictTopDecoded[df.iloc[i]['image']] = tempList

        elif topValues[2] < new_individual_thres:

            tempList = [predictTop[0], predictTop[1], 'new_individual', predictTop[2], predictTop[3]]
            predictTopDecoded[df.iloc[i]['image']] = tempList

        elif topValues[3] < new_individual_thres:

            tempList = [predictTop[0], predictTop[1], predictTop[2], 'new_individual', predictTop[3]]
            predictTopDecoded[df.iloc[i]['image']] = tempList

        elif topValues[4] < new_individual_thres:

            tempList = [predictTop[0], predictTop[1], predictTop[2], predictTop[3], 'new_individual']
            predictTopDecoded[df.iloc[i]['image']] = tempList

        else:
            predictTopDecoded[df.iloc[i]['image']] = predictTop

    for x in tqdm(predictTopDecoded):
        # predictTopDecoded[x] = ' '.join(predictTopDecoded[x])
        # predictTopDecoded[x] = ' '.join(label2name[predictTopDecoded[x]])

        temp = ""
        for i in range(len(predictTopDecoded[x])):
            temp += label2name[predictTopDecoded[x][i]] + " "
        predictTopDecoded[x] = temp
        # for label in predictTopDecoded[x]:

    predictions = pd.Series(predictTopDecoded).reset_index()
    predictions.columns = ['image', 'predictions']
    predictions.to_csv('F:\Dataset\Kaggle\submission_3.csv', index=False)
    predictions.head()





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
        image = self.preprocess(self.data["image"].iloc[idx], idx)
        label = self.labels
        return image, label

    def preprocess(self, image, idx):
        image = os.path.join(self.base_path, image)
        # image = image.split(".")[0] + ".png"
        image = cv2.imread(image)[:, :, ::-1]
        if type(self.data["bbox"].iloc[idx]) is not float:
            bbox = (self.data["bbox"].iloc[idx]).split(" ")
            image = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :]
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
        embeddings_t_new = np.quantile(embeddings_topk["distance"].apply(lambda x: x[0]), 1 - self.q_prior)
        class_t_new = np.quantile(class_topk["distance"].apply(lambda x: x[0]), 1 - self.q_prior)


        def insert_new_individuals(x):
            m = np.array(x["distance"]) > class_t_new
            preds = x["individual"]
            if m.any():
                preds.insert(np.argmax(m), "new_individual")
            preds = preds + [y for y in self.default if y not in preds]
            return preds[:5]

        def insert_new_individuals_1(x):
            m = np.array(x["distance"]) > embeddings_t_new
            preds = x["individual"]
            if m.any():
                preds.insert(np.argmax(m), "new_individual")
            preds = preds + [y for y in self.default if y not in preds]
            return preds[:5]

        preds = class_topk.apply(insert_new_individuals, axis=1)
        preds_1 = embeddings_topk.apply(insert_new_individuals_1, axis=1)
        return preds.values.tolist(), preds_1.values.tolist()


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


class Effnet_Landmark(nn.Module):
    def __init__(self, out_dim, centerperclass=None):
        super(Effnet_Landmark, self).__init__()
        self.feature_extractor = timm.create_model(CFG.MODEL_NAME, pretrained=True)

        # self.pooling = GeM()
        # self.feat = nn.Linear(self.feature_extractor.classifier.in_features, 512)
        self.feat = nn.Linear(self.feature_extractor.classifier.in_features, CFG.EMBEDDING_SIZE)
        # self.swish = Swish_module()
        self.PRelu = torch.nn.PReLU()
        # self.metric_classify = ArcMarginProduct_subcenter(512, out_dim)
        self.metric_classify = ArcMarginProduct_subcenter(CFG.EMBEDDING_SIZE, out_dim, centerperclass)

        self.feature_extractor.classifier = nn.Identity()
        # self.feature_extractor.global_pool = nn.Identity()

        self.bn = nn.BatchNorm1d(CFG.EMBEDDING_SIZE)

    def extract(self, x):
        return self.feature_extractor(x)

    def forward(self, x, label=None):
        x = self.extract(x)
        # x = self.pooling(x).flatten(1)
        # logits_m = self.metric_classify(self.swish(self.feat(x)))
        # return logits_m
        x = self.PRelu(self.bn(self.feat(x)))
        return self.metric_classify(x), F.normalize(x)


class WandDIDNet(LightningModule):
    def __init__(self, margin = None, out_dim=None, centerperclass=None):
        super().__init__()
        self.save_hyperparameters()
        # self.continuous_scheduler = continuous_scheduler
        # Layers
        self.feature_extractor = timm.create_model(CFG.MODEL_NAME, pretrained=True)
        'efficenet'
        in_features = self.feature_extractor.classifier.in_features
        self.feature_extractor.classifier = nn.Identity()
        'seresnet'
        # in_features = self.feature_extractor.fc.in_features
        # self.feature_extractor.fc = nn.Identity()
        self.feature_extractor.global_pool = nn.Identity()
        self.pooling = GeM()
        self.dropout = nn.Dropout()
        self.dense = nn.Linear(in_features, CFG.EMBEDDING_SIZE)
        # self.swish = Swish_module()
        self.metric_classify = ArcMarginProduct_subcenter(CFG.EMBEDDING_SIZE, out_dim, centerperclass)
        # # Loss
        # self.criterion = ArcFaceLossAdaptiveMargin(margins=margin, out_dim=out_dim, s=30)
        # Metrics
        # self.train_acc = torchmetrics.Accuracy()
        # self.train_top_k_acc = torchmetrics.Accuracy(top_k=5)
        # self.val_acc = torchmetrics.Accuracy()
        # self.val_top_k_acc = torchmetrics.Accuracy(top_k=5)

        # self.train_dataloader = dataloader

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
    # BATCH_SIZE = 16
    BATCH_SIZE = 64
    #ACCUMULATE_GRAD_BATCHES = 8
    ACCUMULATE_GRAD_BATCHES = 1
    IMAGE_SIZE = 512
    #IMAGE_SIZE = 224
    NUM_WORKERS = 0
    ### Model
    MODEL_NAME = "efficientnet_b4"
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

'计算余弦相似度'
# def get_cosine_similarity(embeddings):
#     '''Compute cos distance between n embedding vector and itself.'''
#     similarity_matrix = []
#
#     for embed1 in embeddings:
#         similarity_row = []
#         for embed2 in embeddings:
#             similarity_row.append(1 - spatial.distance.cosine(embed1, embed2))
#         similarity_matrix.append(similarity_row)
#
#     return np.array(similarity_matrix, dtype="float32")
#
# cos_matrix = get_cosine_similarity(example_embeds)
# mask = np.zeros_like(cos_matrix)
# mask[np.triu_indices_from(mask)] = True
#
# plot_heatmap(example_paths, cos_matrix, mask)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    '''第1种集成，class_centers 和 feature embedding 做最近邻，这两种没有结合'''
    BASE_PATH = r"F:\Dataset\Kaggle"
    data = pd.read_csv(os.path.join(BASE_PATH, "fold.csv"))
    weights_path = r"F:\Dataset\Kaggle\output\efficientnet_b4_fold0_best_0.369.pth"

    N_CLASSES = len(data["individual_id"].unique())
    individual_mapping = {k: i for i, k in enumerate(data["individual_id"].unique())}
    tmp = np.sqrt(1 / np.sqrt(data['individual_id'].value_counts().loc[list(individual_mapping)].values))
    MARGINS = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * (CFG.MARGIN_MAX - CFG.MARGIN_MIN) + CFG.MARGIN_MIN

    'model'
    # model = WandDIDNet(out_dim=N_CLASSES, centerperclass=CFG.CENTERS_PER_CLASS)
    model = Effnet_Landmark(out_dim=N_CLASSES, centerperclass=CFG.CENTERS_PER_CLASS)
    model.to(device)
    model.load_state_dict(torch.load(weights_path)['model'])

    'train embedding'
    pred_loader = torch.utils.data.DataLoader(
        # WandDIDPred(data, "cropped_train_images", base_path=BASE_PATH),
        WandDIDPred(data, "train_images", base_path=BASE_PATH),
        # WandDIDPred(data, "seg_img", base_path=BASE_PATH),
        batch_size=CFG.BATCH_SIZE,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
        shuffle=False
    )
    preds = inference(model, pred_loader, device)
    preds = torch.stack(preds, dim=0)
    train_data = data.copy()
    train_data["embedding"] = preds.tolist()
    train_data.to_csv(r"F:\Dataset\Kaggle\output\efficientnet_b4_fold0_best_train_embedding.csv")
    # train_data1 = r"F:\Dataset\Kaggle\output\tf_efficientnet_b4_fold0_best_train_embedding.csv"
    # train_data = pd.read_csv(train_data1)

    'test embedding'
    # Prediction on test data
    test_data = pd.read_csv(os.path.join(BASE_PATH, "seg_test.csv"), index_col="image")
    if "inference_image" in test_data.columns:
        test_data["image"] = test_data["inference_image"]
    else:
        test_data["image"] = test_data.index

    test_loader = torch.utils.data.DataLoader(
        # WandDIDPred(test_data, "cropped_test_images", base_path=BASE_PATH),
        WandDIDPred(test_data, "test_images", base_path=BASE_PATH),
        # WandDIDPred(test_data, "seg_img_test", base_path=BASE_PATH),
        batch_size=CFG.BATCH_SIZE,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
        shuffle=False
    )
    preds = inference(model, test_loader, device)
    preds = torch.stack(preds, dim=0)
    test_data["embedding"] = preds.tolist()
    test_data.to_csv(r"F:\Dataset\Kaggle\output\efficientnet_b4_fold0_best_test_embedding.csv")


    # train_data["individual_id_integer"] = train_data["individual_id"].map(individual_mapping).fillna(-1)
    # # train_embeddings = np.array(train_data["embedding"].values.tolist())
    # test_embeddings = np.array(test_data["embedding"].values.tolist())
    # class_centers = model.metric_classify.weight.detach().cpu().numpy()
    #
    # solution = Solution({
    #     "embeddings": train_data,
    #     "class_centers": class_centers
    # }, CFG.Q_NEW, individual_mapping)
    # predictions, predictions_1 = solution.predict(test_embeddings)
    # predictions = pd.Series(predictions, test_data.index, name="predictions").map(lambda x: " ".join(x))
    # predictions_1 = pd.Series(predictions_1, test_data.index, name="predictions").map(lambda x: " ".join(x))
    # predictions.to_csv(r"F:\Dataset\Kaggle\submission1.csv")
    # predictions_1.to_csv(r"F:\Dataset\Kaggle\submission_1.csv")


# --------------------------------------------------------------------------------------------------------------------------------------------------------


    '''第2种集成，Landmark Recognition 2020 Competition Third Place Solution，集成分数'''
    # BASE_PATH = r"F:\Dataset\Kaggle"
    # train_data = pd.read_csv(os.path.join(BASE_PATH, "fold.csv"))
    # weights_path = r"F:\Dataset\Kaggle\output\tf_efficientnet_b4_fold0_best.pth"
    #
    # N_CLASSES = len(train_data["individual_id"].unique())
    # individual_mapping = {k: i for i, k in enumerate(train_data["individual_id"].unique())}
    # tmp = np.sqrt(1 / np.sqrt(train_data['individual_id'].value_counts().loc[list(individual_mapping)].values))
    # MARGINS = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * (CFG.MARGIN_MAX - CFG.MARGIN_MIN) + CFG.MARGIN_MIN
    #
    # pred_loader = torch.utils.data.DataLoader(
    #     # WandDIDPred(data, "train_images", base_path=BASE_PATH),
    #     WandDIDPred(train_data, "seg_img", base_path=BASE_PATH),
    #     batch_size=CFG.BATCH_SIZE,
    #     num_workers=CFG.NUM_WORKERS,
    #     pin_memory=True,
    #     shuffle=False
    # )
    #
    # # Prediction on test data
    # test_data = pd.read_csv(os.path.join(BASE_PATH, "seg_test.csv"), index_col="image")
    # if "inference_image" in test_data.columns:
    #     test_data["image"] = test_data["inference_image"]
    # else:
    #     test_data["image"] = test_data.index
    #
    # test_loader = torch.utils.data.DataLoader(
    #     # WandDIDPred(test_data, "test_images", base_path=BASE_PATH),
    #     WandDIDPred(test_data, "seg_img_test", base_path=BASE_PATH),
    #     batch_size=4,
    #     # batch_size=CFG.BATCH_SIZE,
    #     num_workers=CFG.NUM_WORKERS,
    #     pin_memory=True,
    #     shuffle=False
    # )
    #
    # model = WandDIDNet(margin=MARGINS, out_dim=N_CLASSES, centerperclass=CFG.CENTERS_PER_CLASS)
    # # model = Effnet_Landmark(out_dim=N_CLASSES, centerperclass=CFG.CENTERS_PER_CLASS)
    # model.to(device)
    # model.load_state_dict(torch.load(weights_path)['model'])
    #
    #
    #
    # dir = r"F:\Dataset\Kaggle\before"
    # with open(os.path.join(dir, 'idx2individual_id.pkl'), 'rb') as fp:
    #     idx2landmark_id = pickle.load(fp)
    #     landmark_id2idx = {idx2landmark_id[idx]: idx for idx in idx2landmark_id.keys()}
    #
    # # df = pd.read_csv(os.path.join(dir, 'train.csv'))
    # # pred_mask = pd.Series(df.individual_id.unique()).map(landmark_id2idx).values
    # pred_mask = pd.Series(train_data.individual_id.unique()).map(landmark_id2idx).values
    #
    # model.eval()
    # TOP_K = 5
    # CLS_TOP_K = 5
    # if True:
    #     with torch.no_grad():
    #         feats = []
    #         for image, label in tqdm(pred_loader):  # 672, 768, 512
    #             _, feat_ = model(image.cuda(), label.cuda())
    #
    #
    #             # feat = torch.cat(
    #             #     [feat_b7, feat_b6, feat_b5, feat_b4, feat_b3, feat_nest101, feat_rex2, feat_b6b, feat_b5b], dim=1)
    #             #             print(feat.shape)
    #             feats.append(feat_.detach().cpu())
    #         feats = torch.cat(feats)
    #         feats = feats.cuda()
    #         # feat_ = F.normalize(feat_)
    #         feats = F.normalize(feats)
    #
    #         PRODS = []
    #         PREDS = []
    #         PRODS_M = []
    #         PREDS_M = []
    #         for image, label in tqdm(test_loader):
    #
    #             # probs_m = torch.zeros([4, 81313], device=device)
    #             probs_m = torch.zeros([4, N_CLASSES], device=device)
    #
    #             logits_m, feat_ = model(image.cuda(), label.cuda())
    #             probs_m += logits_m                                                 # logits_m 是arcface的得分
    #             # feat = torch.cat(
    #             #     [feat_b7, feat_b6, feat_b5, feat_b4, feat_b3, feat_nest101, feat_rex2, feat_b6b, feat_b5b], dim=1)
    #             feat_ = F.normalize(feat_)
    #
    #
    #             probs_m[:, pred_mask] += 1.0
    #             probs_m -= 1.0
    #
    #             (values, indices) = torch.topk(probs_m, CLS_TOP_K, dim=1)
    #             probs_m = values
    #             preds_m = indices
    #             PRODS_M.append(probs_m.detach().cpu())  # 模型输出得分
    #             PREDS_M.append(preds_m.detach().cpu())
    #
    #             distance = feat_.mm(feats.t())
    #             (values, indices) = torch.topk(distance, TOP_K, dim=1)
    #             probs = values
    #             preds = indices
    #             PRODS.append(probs.detach().cpu())  # 余弦相似度
    #             PREDS.append(preds.detach().cpu())
    #
    #         PRODS = torch.cat(PRODS).numpy()
    #         PREDS = torch.cat(PREDS).numpy()
    #         PRODS_M = torch.cat(PRODS_M).numpy()
    #         PREDS_M = torch.cat(PREDS_M).numpy()
    #
    # # map both to landmark_id
    # gallery_landmark = train_data['individual_id'].values
    # PREDS = gallery_landmark[PREDS]
    # PREDS_M = np.vectorize(idx2landmark_id.get)(PREDS_M)
    #
    # PRODS_F = []
    # PREDS_F = []
    # for i in tqdm(range(PREDS.shape[0])):
    #     tmp = {}
    #     classify_dict = {PREDS_M[i, j]: PRODS_M[i, j] for j in range(CLS_TOP_K)}
    #     for k in range(TOP_K):
    #         lid = PREDS[i, k]
    #         # tmp[lid] = tmp.get(lid, 0.) + float(PRODS[i, k]) ** 9 * classify_dict.get(lid, 1e-8) ** 10
    #         tmp[lid] = tmp.get(lid, 0.) + float(PRODS[i, k]) ** 8 * classify_dict.get(lid, 1e-8) ** 8
    #     pred, conf = max(tmp.items(), key=lambda x: x[1])
    #     PREDS_F.append(pred)
    #     PRODS_F.append(conf)
    # test_data['predictions'] =
    # print(1)
    # df_test['pred_id'] = PREDS_F
    # df_test['pred_conf'] = PRODS_F
    #
    # df_sub['landmarks'] = df_test.apply(lambda row: f'{row["pred_id"]} {row["pred_conf"]}', axis=1)
    # df_sub.to_csv('submission.csv', index=False)
# --------------------------------------------------------------------------------------------------------------------------------------------------------

    '''第3种集成，预测前两项必有new_individual，不太合理'''
    BASE_PATH = r"F:\Dataset\Kaggle"
    data = pd.read_csv(os.path.join(BASE_PATH, "fold.csv"))
    # weights_path = r"F:\Dataset\Kaggle\output\tf_efficientnet_b4_fold0_best.pth"
    #
    # N_CLASSES = len(data["individual_id"].unique())
    individual_mapping = {k: i for i, k in enumerate(data["individual_id"].unique())}

    'model'
    # # model = WandDIDNet(margin=MARGINS, out_dim=N_CLASSES, centerperclass=CFG.CENTERS_PER_CLASS)
    # model = Effnet_Landmark(out_dim=N_CLASSES, centerperclass=CFG.CENTERS_PER_CLASS)
    # model.to(device)
    # model.load_state_dict(torch.load(weights_path)['model'])

    'train embedding'
    # pred_loader = torch.utils.data.DataLoader(
    #     # WandDIDPred(data, "train_images", base_path=BASE_PATH),
    #     WandDIDPred(data, "seg_img", base_path=BASE_PATH),
    #     batch_size=CFG.BATCH_SIZE,
    #     num_workers=CFG.NUM_WORKERS,
    #     pin_memory=True,
    #     shuffle=False
    # )
    # preds = inference(model, pred_loader, device)
    # preds = torch.stack(preds, dim=0)
    # train_data = data.copy()
    # train_data["embedding"] = preds.tolist()
    # train_data.to_csv(r"F:\Dataset\Kaggle\output\tf_efficientnet_b4_fold0_best_train_embedding.csv")
    train_data1 = r"F:\Dataset\Kaggle\output\efficientnet_b4_fold0_best_train_embedding.csv"
    train_data = pd.read_csv(train_data1)
    train_embedding = np.zeros([train_data.shape[0], CFG.EMBEDDING_SIZE])
    for i in range(len(train_data.loc[:, "embedding"].values)):
        train_embedding[i] = np.array(list(map(float, train_data.loc[:, "embedding"].values[i][1:][:-1].split(", "))))

    'test embedding'
    # # Prediction on test data
    # test_data = pd.read_csv(os.path.join(BASE_PATH, "seg_test.csv"), index_col="image")
    # if "inference_image" in test_data.columns:
    #     test_data["image"] = test_data["inference_image"]
    # else:
    #     test_data["image"] = test_data.index
    #
    # test_loader = torch.utils.data.DataLoader(
    #     # WandDIDPred(test_data, "test_images", base_path=BASE_PATH),
    #     WandDIDPred(test_data, "seg_img_test", base_path=BASE_PATH),
    #     batch_size=CFG.BATCH_SIZE,
    #     num_workers=CFG.NUM_WORKERS,
    #     pin_memory=True,
    #     shuffle=False
    # )
    # preds = inference(model, test_loader, device)
    # preds = torch.stack(preds, dim=0)
    # test_data["embedding"] = preds.tolist()
    # test_data.to_csv(r"F:\Dataset\Kaggle\output\tf_efficientnet_b4_fold0_best_test_embedding.csv")
    #
    # train_data["individual_id_integer"] = train_data["individual_id"].map(individual_mapping).fillna(-1)
    # train_embeddings = np.array(train_data["embedding"].values.tolist())
    # test_embeddings = np.array(test_data["embedding"].values.tolist())
    test_data1 = r"F:\Dataset\Kaggle\output\efficientnet_b4_fold0_best_test_embedding.csv"
    test_data = pd.read_csv(test_data1)
    test_embedding = np.zeros([test_data.shape[0], CFG.EMBEDDING_SIZE])
    for i in range(len(test_data.loc[:, "embedding"].values)):
        test_embedding[i] = np.array(list(map(float, test_data.loc[:, "embedding"].values[i][1:][:-1].split(", "))))

    # PRETRAINED_NAME1 = "EffNetB0_fold_0_loss_14.979"
    # PRETRAINED_NAME2 = "EffNetB0_fold_1_loss_14.91"
    # PRETRAINED_NAME3 = "EffNetB0_fold_2_loss_15.325"
    # MODEL_NAME = 'efficientnet_b0'
    # NUM_CLASSES = 15587
    # NO_NEURONS = 250
    # EMBEDDING_SIZE = 128

    train_individual_ids = train_data["individual_id_integer"].values


    knn_final_model = NearestNeighbors(n_neighbors=50)
    knn_final_model.fit(train_embedding)
    # knn_final_model.fit(train_embeddings)

    D, I = knn_final_model.kneighbors(test_embedding)
    # D, I = knn_final_model.kneighbors(test_embeddings)

    # List of the test dataframe image ids (to loop through it)
    test_images = test_data["image"].tolist()

    test_df = []

    # Loop through each observation within test data
    for k, image_id in tqdm(enumerate(test_images)):
        # Get individual_id & distances for the observation
        individual_id = train_individual_ids[I[k]]
        distances = D[k]
        # Create a df subset with this info
        subset_preds = pd.DataFrame(np.stack([individual_id, distances], axis=1),
                                    columns=['individual_id_integer', 'distances'])
        subset_preds['image_id'] = image_id
        test_df.append(subset_preds)

    # Concatenate subset dataframes into 1 dataframe
    test_df = pd.concat(test_df).reset_index(drop=True)
    # Choose max distance for each unique pair of individual_id & image_id
    test_df = test_df.groupby(['image_id', 'individual_id_integer'])['distances'].max().reset_index()

    predictions = {}
    thresh = 5

    for k, row in tqdm(test_df.iterrows()):
        image_id = row["image_id"]
        individual_id = row["individual_id_integer"]
        distance = row["distances"]

        # If the image_id has already been added in predictions before
        if image_id in predictions:
            # If total preds for this image_id are < 5 then add, else continue
            if len(predictions[image_id]) != 5:
                predictions[image_id].append(individual_id)
            else:
                continue
        # If the distance is greater than thresh add prediction + "new_individual"
        elif distance > thresh:
            predictions[image_id] = [individual_id, "new_individual"]
        else:
            predictions[image_id] = ["new_individual", individual_id]

    # Fill in all lists that have less than 5 predictions as of yet
    sample_list = ['37c7aba965a5', '114207cab555', 'a6e325d8e924', '19fbb960f07d', 'c995c043c353']

    for image_id, preds in tqdm(predictions.items()):
        if len(preds) < 5:
            remaining = [individ_id for individ_id in sample_list if individ_id not in preds]
            preds.extend(remaining)
            predictions[image_id] = preds[:5]

    predictions = pd.Series(predictions).reset_index()
    predictions.columns = ['image', 'predictions']
    predictions['predictions'] = predictions['predictions'].apply(lambda x: ' '.join(x))
    predictions.to_csv(r"F:\Dataset\Kaggle\submission3.csv", index=False)

# -----------------------------------------------------------------------------------------------------------------------------------------------

    '''第4种集成，val embedding 挑出 new_individual 阈值，再与 test embedding 做最近邻'''
    # BASE_PATH = r"F:\Dataset\Kaggle"
    # data = pd.read_csv(os.path.join(BASE_PATH, "fold.csv"))
    # weights_path = r"F:\Dataset\Kaggle\output\gluon_seresnext101_32x4d_fold0_best.pth"
    #
    # N_CLASSES = len(data["individual_id"].unique())
    # individual_mapping = {k: i for i, k in enumerate(data["individual_id"].unique())}
    # tmp = np.sqrt(1 / np.sqrt(data['individual_id'].value_counts().loc[list(individual_mapping)].values))
    # MARGINS = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * (CFG.MARGIN_MAX - CFG.MARGIN_MIN) + CFG.MARGIN_MIN
    #
    # 'model'
    # model = WandDIDNet(out_dim=N_CLASSES, centerperclass=CFG.CENTERS_PER_CLASS)
    # # model = Effnet_Landmark(out_dim=N_CLASSES, centerperclass=CFG.CENTERS_PER_CLASS)
    # model.to(device)
    # model.load_state_dict(torch.load(weights_path)['model'])
    #
    #
    # trn_idx = data[data['fold'] != 0].index
    # val_idx = data[data['fold'] == 0].index
    #
    # train_folds = data.loc[trn_idx].reset_index(drop=True)
    # valid_folds = data.loc[val_idx].reset_index(drop=True)
    #
    # # predict first on train dataset to extract embeddings
    # train_dataset = WandDIDPred(train_folds, "train_images", base_path=BASE_PATH)
    # train_loader = torch.utils.data.DataLoader(train_dataset,
    #                                            batch_size=CFG.BATCH_SIZE,
    #                                            num_workers=CFG.NUM_WORKERS,
    #                                            pin_memory=True,
    #                                            shuffle=False)
    #
    # valid_dataset = WandDIDPred(valid_folds, "train_images", base_path=BASE_PATH)
    # valid_loader = torch.utils.data.DataLoader(valid_dataset,
    #                           batch_size=CFG.BATCH_SIZE,
    #                           num_workers=CFG.NUM_WORKERS,
    #                           pin_memory=True,
    #                           shuffle=False)
    # valid_dataset = WandDIDPred(valid_folds, "train_images", base_path=BASE_PATH)
    #
    # test_dataset = WandDIDPred(df_test, transforms=data_transforms["test"])
    # test_loader = DataLoader(test_dataset, batch_size=CONFIG['test_batch_size'],
    #                          num_workers=1, shuffle=False, pin_memory=True)
    #
    # 'train embedding'
    # pred_loader = torch.utils.data.DataLoader(
    #     WandDIDPred(train_folds, "train_images", base_path=BASE_PATH),
    #     # WandDIDPred(data, "seg_img", base_path=BASE_PATH),
    #     batch_size=CFG.BATCH_SIZE,
    #     num_workers=CFG.NUM_WORKERS,
    #     pin_memory=True,
    #     shuffle=False
    # )
    # preds = inference(model, pred_loader, device)
    # preds = torch.stack(preds, dim=0)
    # train_data = data.copy()
    # train_data["embedding"] = preds.tolist()
    # train_data.to_csv(r"F:\Dataset\Kaggle\output\tf_efficientnet_b4_fold0_best_train_embedding.csv")
    # # train_data1 = r"F:\Dataset\Kaggle\output\tf_efficientnet_b4_fold0_best_train_embedding.csv"
    # # train_data = pd.read_csv(train_data1)
    #
    # 'test embedding'
    # # Prediction on test data
    # test_data = pd.read_csv(os.path.join(BASE_PATH, "seg_test.csv"), index_col="image")
    # if "inference_image" in test_data.columns:
    #     test_data["image"] = test_data["inference_image"]
    # else:
    #     test_data["image"] = test_data.index
    #
    # test_loader = torch.utils.data.DataLoader(
    #     # WandDIDPred(test_data, "test_images", base_path=BASE_PATH),
    #     WandDIDPred(test_data, "seg_img_test", base_path=BASE_PATH),
    #     batch_size=CFG.BATCH_SIZE,
    #     num_workers=CFG.NUM_WORKERS,
    #     pin_memory=True,
    #     shuffle=False
    # )
    # preds = inference(model, test_loader, device)
    # preds = torch.stack(preds, dim=0)
    # test_data["embedding"] = preds.tolist()
    # test_data.to_csv(r"F:\Dataset\Kaggle\output\tf_efficientnet_b4_fold0_best_test_embedding.csv")



    # train_data1 = r"F:\Dataset\Kaggle\output\efficientnet_b4_fold0_best_train_embedding.csv"
    # train_data = pd.read_csv(train_data1)
    # label2name = {}
    # for i in range(len(train_data['individual_id'].unique())):
    #     label2name[i] = train_data['individual_id'].unique()[i]
    # label2name["new_individual"] = "new_individual"
    #
    # df_train_cnn = train_data[train_data.fold != 0].reset_index(drop=True)
    # df_valid_cnn = train_data[train_data.fold == 0].reset_index(drop=True)
    #
    # train_embedding = np.zeros([df_train_cnn.shape[0], CFG.EMBEDDING_SIZE])
    # for i in range(len(df_train_cnn.loc[:, "embedding"].values)):
    #     train_embedding[i] = np.array(list(map(float, df_train_cnn.loc[:, "embedding"].values[i][1:][:-1].split(", "))))
    # train_cnn_labels = np.array(df_train_cnn['individual_id_integer'].values)
    #
    # val_embedding = np.zeros([df_valid_cnn.shape[0], CFG.EMBEDDING_SIZE])
    # for i in range(len(df_valid_cnn.loc[:, "embedding"].values)):
    #     val_embedding[i] = np.array(list(map(float, df_valid_cnn.loc[:, "embedding"].values[i][1:][:-1].split(", "))))
    # valid_cnn_labels = np.array(df_valid_cnn['individual_id_integer'].values)
    #
    # test_data1 = r"F:\Dataset\Kaggle\output\efficientnet_b4_fold0_best_test_embedding.csv"
    # test_data = pd.read_csv(test_data1)
    # test_embedding = np.zeros([test_data.shape[0], CFG.EMBEDDING_SIZE])
    # for i in range(len(test_data.loc[:, "embedding"].values)):
    #     test_embedding[i] = np.array(list(map(float, test_data.loc[:, "embedding"].values[i][1:][:-1].split(", "))))
    #
    # iteration = 0
    # best_score = 0
    # best_thres = 0
    # for thres in np.arange(0.1, 0.9, 0.1):
    #     print("iteration ", iteration, " of ", len(np.arange(0.3, 0.9, 0.1)))
    #     iteration += 1
    #     score = PredictGrid(train_embedding, val_embedding, train_cnn_labels, valid_cnn_labels,
    #                         new_individual_thres=thres)
    #     if (score > best_score):
    #         best_score = score
    #         best_thres = thres
    #     print("thres: ", thres, ",score: ", score)
    # print("Best score is: ", best_score)
    # print("Best thres is: ", best_thres)
    #
    # # test_cnn_predictions = np.array(inference(model, test_loader, CONFIG['device']))
    # allTrainData = np.concatenate((train_embedding, val_embedding))
    # allTrainingLabels = np.concatenate((train_cnn_labels, valid_cnn_labels))
    # GetSubmission(allTrainData, test_embedding, allTrainingLabels, neighbors=CFG.KNN, metric='cosine',
    #               new_individual_thres=best_thres, label2name=label2name)


    '''第5种集成：结合 1 2 4，to do'''
    # train_data1 = r"F:\Dataset\Kaggle\output\tf_efficientnet_b4_fold0_best_train_embedding.csv"
    # train_data = pd.read_csv(train_data1)
    # N_CLASSES = len(train_data["individual_id"].unique())
    # label2name = {}
    # for i in range(len(train_data['individual_id'].unique())):
    #     label2name[i] = train_data['individual_id'].unique()[i]
    # label2name["new_individual"] = "new_individual"
    # individual_mapping = {k: i for i, k in enumerate(train_data["individual_id"].unique())}
    #
    # df_train_cnn = train_data[train_data.fold != 0].reset_index(drop=True)
    # df_valid_cnn = train_data[train_data.fold == 0].reset_index(drop=True)
    #
    # train_embedding = np.zeros([df_train_cnn.shape[0], CFG.EMBEDDING_SIZE])
    # for i in range(len(df_train_cnn.loc[:, "embedding"].values)):
    #     train_embedding[i] = np.array(list(map(float, df_train_cnn.loc[:, "embedding"].values[i][1:][:-1].split(", "))))
    # train_cnn_labels = np.array(df_train_cnn['individual_id_integer'].values)
    #
    # val_embedding = np.zeros([df_valid_cnn.shape[0], CFG.EMBEDDING_SIZE])
    # for i in range(len(df_valid_cnn.loc[:, "embedding"].values)):
    #     val_embedding[i] = np.array(list(map(float, df_valid_cnn.loc[:, "embedding"].values[i][1:][:-1].split(", "))))
    # valid_cnn_labels = np.array(df_valid_cnn['individual_id_integer'].values)
    #
    # test_data1 = r"F:\Dataset\Kaggle\output\tf_efficientnet_b4_fold0_best_test_embedding.csv"
    # test_data = pd.read_csv(test_data1)
    # test_embedding = np.zeros([test_data.shape[0], CFG.EMBEDDING_SIZE])
    # for i in range(len(test_data.loc[:, "embedding"].values)):
    #     test_embedding[i] = np.array(list(map(float, test_data.loc[:, "embedding"].values[i][1:][:-1].split(", "))))
    #
    # weights_path = r"F:\Dataset\Kaggle\output\tf_efficientnet_b4_fold0_best.pth"
    # model = WandDIDNet(out_dim=N_CLASSES, centerperclass=CFG.CENTERS_PER_CLASS)
    # # model = Effnet_Landmark(out_dim=N_CLASSES, centerperclass=CFG.CENTERS_PER_CLASS)
    # model.to(device)
    # model.load_state_dict(torch.load(weights_path)['model'])
    #
    # class_centers = model.metric_classify.weight.detach().cpu().numpy()
    #
    #
    # iteration = 0
    # best_score = 0
    # best_thres = 0
    # for thres in np.arange(0.1, 0.9, 0.1):
    #     print("iteration ", iteration, " of ", len(np.arange(0.3, 0.9, 0.1)))
    #     iteration += 1
    #     score = PredictGrid(train_embedding, val_embedding, train_cnn_labels, valid_cnn_labels,
    #                         new_individual_thres=thres, class_centers=class_centers, individual_mapping=individual_mapping)
    #     if (score > best_score):
    #         best_score = score
    #         best_thres = thres
    #     print("thres: ", thres, ",score: ", score)
    # print("Best score is: ", best_score)
    # print("Best thres is: ", best_thres)
    #
    # # test_cnn_predictions = np.array(inference(model, test_loader, CONFIG['device']))
    # allTrainData = np.concatenate((train_embedding, val_embedding))
    # allTrainingLabels = np.concatenate((train_cnn_labels, valid_cnn_labels))
    # GetSubmission(allTrainData, test_embedding, allTrainingLabels, neighbors=CFG.KNN, metric='cosine',
    #               new_individual_thres=best_thres, label2name=label2name)


    '第6种集成，Landmark Recognition 2020 Competition One Place Solution，惩罚相似度'

    # train_df = pd.read_csv('../input/landmark-recognition-2021/train.csv')
    #
    # if len(train_df) == 1580470:  # submission use all the training images
    #     records = {}
    #
    #     for image_id, landmark_id in train_df.values:
    #         if landmark_id in records:
    #             records[landmark_id].append(image_id)
    #         else:
    #             records[landmark_id] = [image_id]
    #
    #     image_ids = []
    #     landmark_ids = []
    #
    #     for landmark_id, img_ids in records.items():
    #         num = min(len(img_ids), 2)  # maxium two images
    #         image_ids.extend(records[landmark_id][:num])
    #         landmark_ids.extend([landmark_id] * num)
    #
    #     train_df = pd.DataFrame({'id': image_ids, 'landmark_id': landmark_ids})
    # # train_df = train_df.iloc[:512,]
    # train_df.to_csv(TRAIN_LABEL_FILE, index=False)
    # train_df
    #
    # @torch.inference_mode()
    # def get_embeddings(
    #         module: pl.LightningModule, dataloader: DataLoader, encoder: LabelEncoder, stage: str
    # ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #
    #     all_image_names = []
    #     all_embeddings = []
    #     all_targets = []
    #
    #     for batch in tqdm(dataloader, desc=f"Creating {stage} embeddings"):
    #         image_names = batch["image_name"]
    #         images = batch["image"].to(module.device)
    #         targets = batch["target"].to(module.device)
    #
    #         embeddings = module(images)
    #
    #         all_image_names.append(image_names)
    #         all_embeddings.append(embeddings.cpu().numpy())
    #         all_targets.append(targets.cpu().numpy())
    #
    #     all_image_names = np.concatenate(all_image_names)
    #     all_embeddings = np.vstack(all_embeddings)
    #     all_targets = np.concatenate(all_targets)
    #
    #     all_embeddings = normalize(all_embeddings, axis=1, norm="l2")
    #     all_targets = encoder.inverse_transform(all_targets)
    #
    #     return all_image_names, all_embeddings, all_targets
    #
    # def create_and_search_index(embedding_size: int, train_embeddings: np.ndarray, val_embeddings: np.ndarray, k: int):
    #     index = faiss.IndexFlatIP(embedding_size)
    #     index.add(train_embeddings)
    #     D, I = index.search(val_embeddings, k=k)  # noqa: E741
    #
    #     return D, I
    #
    # def create_val_targets_df(
    #         train_targets: np.ndarray, val_image_names: np.ndarray, val_targets: np.ndarray
    # ) -> pd.DataFrame:
    #
    #     allowed_targets = np.unique(train_targets)
    #     val_targets_df = pd.DataFrame(np.stack([val_image_names, val_targets], axis=1), columns=["image", "target"])
    #     val_targets_df.loc[~val_targets_df.target.isin(allowed_targets), "target"] = "new_individual"
    #
    #     return val_targets_df
    #
    # def create_distances_df(
    #         image_names: np.ndarray, targets: np.ndarray, D: np.ndarray, I: np.ndarray, stage: str  # noqa: E741
    # ) -> pd.DataFrame:
    #
    #     distances_df = []
    #     for i, image_name in tqdm(enumerate(image_names), desc=f"Creating {stage}_df"):
    #         target = targets[I[i]]
    #         distances = D[i]
    #         subset_preds = pd.DataFrame(np.stack([target, distances], axis=1), columns=["target", "distances"])
    #         subset_preds["image"] = image_name
    #         distances_df.append(subset_preds)
    #
    #     distances_df = pd.concat(distances_df).reset_index(drop=True)
    #     distances_df = distances_df.groupby(["image", "target"]).distances.max().reset_index()
    #     distances_df = distances_df.sort_values("distances", ascending=False).reset_index(drop=True)
    #
    #     return distances_df
    #
    # def get_best_threshold(val_targets_df: pd.DataFrame, valid_df: pd.DataFrame) -> Tuple[float, float]:
    #     best_th = 0
    #     best_cv = 0
    #     for th in [0.1 * x for x in range(11)]:
    #         all_preds = get_predictions(valid_df, threshold=th)
    #
    #         cv = 0
    #         for i, row in val_targets_df.iterrows():
    #             target = row.target
    #             preds = all_preds[row.image]
    #             val_targets_df.loc[i, th] = map_per_image(target, preds)
    #
    #         cv = val_targets_df[th].mean()
    #
    #         print(f"th={th} cv={cv}")
    #
    #         if cv > best_cv:
    #             best_th = th
    #             best_cv = cv
    #
    #     print(f"best_th={best_th}")
    #     print(f"best_cv={best_cv}")
    #
    #     # Adjustment: Since Public lb has nearly 10% 'new_individual' (Be Careful for private LB)
    #     val_targets_df["is_new_individual"] = val_targets_df.target == "new_individual"
    #     val_scores = val_targets_df.groupby("is_new_individual").mean().T
    #     val_scores["adjusted_cv"] = val_scores[True] * 0.1 + val_scores[False] * 0.9
    #     best_th = val_scores["adjusted_cv"].idxmax()
    #     print(f"best_th_adjusted={best_th}")
    #
    #     return best_th, best_cv
    #
    # def get_predictions(df: pd.DataFrame, threshold: float = 0.2):
    #     sample_list = ["938b7e931166", "5bf17305f073", "7593d2aee842", "7362d7a01d00", "956562ff2888"]
    #
    #     predictions = {}
    #     for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Creating predictions for threshold={threshold}"):
    #         if row.image in predictions:
    #             if len(predictions[row.image]) == 5:
    #                 continue
    #             predictions[row.image].append(row.target)
    #         elif row.distances > threshold:
    #             predictions[row.image] = [row.target, "new_individual"]
    #         else:
    #             predictions[row.image] = ["new_individual", row.target]
    #
    #     for x in tqdm(predictions):
    #         if len(predictions[x]) < 5:
    #             remaining = [y for y in sample_list if y not in predictions]
    #             predictions[x] = predictions[x] + remaining
    #             predictions[x] = predictions[x][:5]
    #
    #     return predictions
    #
    # # TODO: add types
    # def map_per_image(label, predictions):
    #     """Computes the precision score of one image.
    #
    #     Parameters
    #     ----------
    #     label : string
    #             The true label of the image
    #     predictions : list
    #             A list of predicted elements (order does matter, 5 predictions allowed per image)
    #
    #     Returns
    #     -------
    #     score : double
    #     """
    #     try:
    #         return 1 / (predictions[:5].index(label) + 1)
    #     except ValueError:
    #         return 0.0
    #
    # def create_predictions_df(test_df: pd.DataFrame, best_th: float) -> pd.DataFrame:
    #     predictions = get_predictions(test_df, best_th)
    #
    #     predictions = pd.Series(predictions).reset_index()
    #     predictions.columns = ["image", "predictions"]
    #     predictions["predictions"] = predictions["predictions"].apply(lambda x: " ".join(x))
    #
    #     return predictions
    #
    # def infer(
    #         checkpoint_path: str,
    #         train_csv_encoded_folded: str = str(TRAIN_CSV_ENCODED_FOLDED_PATH),
    #         test_csv: str = str(TEST_CSV_PATH),
    #         val_fold: float = 0.0,
    #         image_size: int = 256,
    #         batch_size: int = 64,
    #         num_workers: int = 2,
    #         k: int = 50,
    # ):
    #     module = load_eval_module(checkpoint_path, torch.device("cuda"))
    #
    #     train_dl, val_dl, test_dl = load_dataloaders(
    #         train_csv_encoded_folded=train_csv_encoded_folded,
    #         test_csv=test_csv,
    #         val_fold=val_fold,
    #         image_size=image_size,
    #         batch_size=batch_size,
    #         num_workers=num_workers,
    #     )
    #
    #     encoder = load_encoder()
    #
    #     train_image_names, train_embeddings, train_targets = get_embeddings(module, train_dl, encoder, stage="train")
    #     val_image_names, val_embeddings, val_targets = get_embeddings(module, val_dl, encoder, stage="val")
    #     test_image_names, test_embeddings, test_targets = get_embeddings(module, test_dl, encoder, stage="test")
    #
    #     D, I = create_and_search_index(module.hparams.embedding_size, train_embeddings, val_embeddings, k)  # noqa: E741
    #     print("Created index with train_embeddings")
    #
    #     val_targets_df = create_val_targets_df(train_targets, val_image_names, val_targets)
    #     print(f"val_targets_df=\n{val_targets_df.head()}")
    #
    #     val_df = create_distances_df(val_image_names, train_targets, D, I, "val")
    #     print(f"val_df=\n{val_df.head()}")
    #
    #     best_th, best_cv = get_best_threshold(val_targets_df, val_df)
    #     print(f"val_targets_df=\n{val_targets_df.describe()}")
    #
    #     train_embeddings = np.concatenate([train_embeddings, val_embeddings])
    #     train_targets = np.concatenate([train_targets, val_targets])
    #     print("Updated train_embeddings and train_targets with val data")
    #
    #     D, I = create_and_search_index(module.hparams.embedding_size, train_embeddings, test_embeddings,
    #                                    k)  # noqa: E741
    #     print("Created index with train_embeddings")
    #
    #     test_df = create_distances_df(test_image_names, train_targets, D, I, "test")
    #     print(f"test_df=\n{test_df.head()}")
    #
    #     predictions = create_predictions_df(test_df, best_th)
    #     print(f"predictions.head()={predictions.head()}")
    #     predictions.to_csv(SUBMISSION_CSV_PATH, index=False)



if __name__ == "__main__":
    main()