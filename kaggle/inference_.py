import cv2
import numpy as np
import torch
# from .train_ import CustomResNext, TrainDataset
from kaggle.train1_ import WandDIDNet
from kaggle.train_ import GeM, ArcMarginProduct
import timm
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import pandas as pd
from albumentations.pytorch import ToTensorV2
import albumentations as A

from sklearn.neighbors import NearestNeighbors


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
    return np.mean([map_per_image(l, p) for l,p in zip(labels, predictions)])


def PredictGrid(train_cnn_predictions, valid_cnn_predictions, train_labels, valid_labels, new_individual_thres):
    neigh = NearestNeighbors(n_neighbors=CONFIG["neigh"], metric="cosine")
    neigh.fit(train_cnn_predictions)

    distances, idxs = neigh.kneighbors(valid_cnn_predictions, return_distance=True)
    conf = 1 - distances
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
        predictTopDecoded[x] = ' '.join(predictTopDecoded[x])

        # for label in predictTopDecoded[x]:

    predictions = pd.Series(predictTopDecoded).reset_index()
    predictions.columns = ['image', 'predictions']
    predictions.to_csv('submission.csv', index=False)
    predictions.head()


@torch.inference_mode()
def inference(model, dataloader, device):
    model.eval()
    outputList=[]
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data['image'].to(device, dtype=torch.float)
        labels = data['label'].to(device, dtype=torch.long)
        _, outputs = model(images,labels)
        outputList.extend(outputs.cpu().detach().numpy())
    return outputList


class HappyWhaleDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.file_names = df['file_path'].values
        self.labels = df['dummy_labels'].values
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[index]

        if self.transforms:
            img = self.transforms(image=img)["image"]
        return {
            'image': img,
            'label': torch.tensor(label, dtype=torch.long)
        }

class HappyWhaleModel(nn.Module):
    def __init__(self, model_name, pretrained=True):
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
        self.embedding = nn.Sequential(nn.Linear(in_features, CONFIG["embedding_size"]),
                                       nn.BatchNorm1d(CONFIG["embedding_size"]),
                                       nn.PReLU())
                                       # nn.Dropout(p=0.2),
                                       # nn.Linear(CFG.noNeurons, CFG.embedding_size),
                                       # nn.BatchNorm1d(CFG.embedding_size),
                                       # nn.PReLU(num_parameters=1, init=0.25),
                                       # nn.Dropout(p=0.2))
        self.arcface = ArcMarginProduct(CONFIG["embedding_size"],
                                   CONFIG["num_classes"],
                                   s=CONFIG["s"],
                                   m=CONFIG["m"],
                                   easy_margin=CONFIG["easy_margin"],
                                   ls_eps=CONFIG["ls_eps"],
                                   device=CONFIG['device'])


    def forward(self, images, labels):
        # features = self.model(images)
        # pooled_features = self.pooling(features).flatten(1)
        # # pooled_drop = self.drop(pooled_features)
        # emb = self.fc(pooled_features)
        # output = self.arc(emb,labels)
        #
        #
        # return output,emb

        x = self.model(images)
        # return x
        gem_pool = self.pooling(x).flatten(1)
        embedding = self.embedding(gem_pool)
        pred = self.arcface(embedding, labels)
        return pred, embedding








def main(CONFIG):
    def get_test_file_path(id):
        return f"{TEST_DIR}/{id}"

    def get_train_file_path(id):
        return f"{TRAIN_DIR}/{id}"

    ROOT_DIR = r'F:\Dataset\Kaggle'
    TEST_DIR = r'F:\Dataset\Kaggle\test_images'
    TRAIN_DIR = r'F:\Dataset\Kaggle\train_images'
    weights_path = r"F:\Dataset\Kaggle\output\tf_efficientnet_b4_fold0_best.pth"





    data_transforms = {
        "test": A.Compose([
            A.Resize(CONFIG['img_size'], CONFIG['img_size']),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2()], p=1.)
    }

    if CONFIG["test_mode"] == True:
        df_test = pd.read_csv(f"{ROOT_DIR}/sample_submission.csv")[:2000]
        df_train = pd.read_csv(f"{ROOT_DIR}/train.csv")[:2000]
    else:
        df_test = pd.read_csv(f"{ROOT_DIR}/sample_submission.csv")
        df_train = pd.read_csv(f"{ROOT_DIR}/train.csv")

    # train_file = pd.read_csv(r'F:\Dataset\Kaggle\train.csv')
    # # train_file = pd.read_csv(r'F:\Dataset\Kaggle\train2.csv')
    # # PreProcess
    # df_train['species'].replace({
    #     'bottlenose_dolpin': 'bottlenose_dolphin',
    #     'kiler_whale': 'killer_whale',
    #     'beluga': 'beluga_whale',
    #     'globis': 'short_finned_pilot_whale',
    #     'pilot_whale': 'short_finned_pilot_whale'
    # }, inplace=True)
    #
    # label2name = {}
    # for i in range(len(df_train['individual_id'].unique())):
    #     label2name[i] = df_train['individual_id'].unique()[i]
    # name2label = dict((v, k) for k, v in label2name.items())
    # df_train['individual_id'].replace(name2label, inplace=True)



    df_test['file_path'] = df_test['image'].apply(get_test_file_path)
    df_train['file_path'] = df_train['image'].apply(get_train_file_path)
    train_labels = np.array(df_train['individual_id'].values)
    # split into train and valid like in the training notebook for validating NearestNeighbors approach
    trainFold = 0  # this model was trained on fold 0
    # skf = StratifiedKFold(n_splits=CONFIG['n_fold'], shuffle=True, random_state=CFG.seed)
    Fold = StratifiedKFold(n_splits=CONFIG["n_fold"], shuffle=True, random_state=CONFIG["seed"])
    for fold, (_, val_) in enumerate(Fold.split(X=df_train, y=train_labels)):
        df_train.loc[val_, "kfold"] = fold
    df_train_cnn = df_train[df_train.kfold != trainFold].reset_index(drop=True)
    df_valid_cnn = df_train[df_train.kfold == trainFold].reset_index(drop=True)

    # hardcode dummy label for input in ArcMargin forward function
    df_test['dummy_labels'] = 0
    df_train_cnn['dummy_labels'] = 0
    df_valid_cnn['dummy_labels'] = 0






    model = HappyWhaleModel(CONFIG['model_name'])
    model.to(CONFIG['device'])
    model.load_state_dict(torch.load(weights_path)['model'])



    # predict first on train dataset to extract embeddings
    train_dataset = HappyWhaleDataset(df_train_cnn, transforms=data_transforms["test"])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['test_batch_size'],
                              num_workers=1, shuffle=False, pin_memory=True)

    valid_dataset = HappyWhaleDataset(df_valid_cnn, transforms=data_transforms["test"])
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['test_batch_size'],
                              num_workers=1, shuffle=False, pin_memory=True)

    test_dataset = HappyWhaleDataset(df_test, transforms=data_transforms["test"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['test_batch_size'],
                             num_workers=1, shuffle=False, pin_memory=True)




    df_train_cnn_predictions = np.array(inference(model, train_loader, CONFIG['device']))
    df_valid_cnn_predictions = np.array(inference(model, valid_loader, CONFIG['device']))
    train_cnn_labels = np.array(df_train_cnn['individual_id'].values)
    valid_cnn_labels = np.array(df_valid_cnn['individual_id'].values)

    iteration = 0
    best_score = 0
    best_thres = 0
    for thres in np.arange(0.1, 0.9, 0.1):
        print("iteration ", iteration, " of ", len(np.arange(0.3, 0.9, 0.1)))
        iteration += 1
        score = PredictGrid(df_train_cnn_predictions, df_valid_cnn_predictions, train_cnn_labels, valid_cnn_labels,
                            new_individual_thres=thres)
        if (score > best_score):
            best_score = score
            best_thres = thres
        print("thres: ", thres, ",score: ", score)
    print("Best score is: ", best_score)
    print("Best thres is: ", best_thres)







    test_cnn_predictions = np.array(inference(model, test_loader, CONFIG['device']))
    allTrainData = np.concatenate((df_train_cnn_predictions, df_valid_cnn_predictions))
    allTrainingLabels = np.concatenate((train_cnn_labels, valid_cnn_labels))
    GetSubmission(allTrainData, test_cnn_predictions, allTrainingLabels, neighbors=CONFIG["neigh"], metric='cosine',
                  new_individual_thres=best_thres)
# class CFG:
#     SEED = 69
#     ### Dataset
#     ## Effective batch size will be BATCH_SIZE*ACCUMULATE_GRAD_BATCHES
#     BATCH_SIZE = 16
#     # BATCH_SIZE = 48
#     #ACCUMULATE_GRAD_BATCHES = 8
#     ACCUMULATE_GRAD_BATCHES = 1
#     IMAGE_SIZE = 512
#     #IMAGE_SIZE = 224
#     NUM_WORKERS = 0
#     ### Model
#     model_name = "efficientnet_b0"
#     EMBEDDING_SIZE = 512
#     ### Training
#     ## Arcfaces
#     CENTERS_PER_CLASS = 3
#     S = 30
#     MARGIN_MIN = 0.2
#     MARGIN_MAX = 0.4
#     EPOCHS = 20
#     MIXED_PRECISION = False
#     MODEL_PATH = r"F:\Dataset\Kaggle\output"
#     # Inference
#     KNN = 100
#     Q_NEW = 0.112 # Proportion of new individuals expected in the dataset
#
#     continuous_scheduler = True
#     print_freq = 100


if __name__ == "__main__":
    CONFIG = {"seed": 42,  # choose your lucky seed
              "img_size": 762,  # training image size
              "model_name": "tf_efficientnet_b4",  # training model arhitecture
              "num_classes": 15587,  # total individuals in training data
              "test_batch_size": 32,  # choose acording to the training arhitecture and image size
              "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),  # gpu
              "test_mode": True,
              # selects just the first 2000 samples from the test data, usefull for debuging purposes
              "percentage_new_from_test": 10,  # how much of the test data is estimated to be "new_individual"
              "threshold": 0.1,  # it will be overwriten after prediction histogram
              "neigh": 100,  # knn neighbors
              "n_fold": 5,  # nr of folds that the model has been trained
              "embedding_size": 512,
              # ArcFace Hyperparameters
              "s": 30.0,
              "m": 0.50,
              "ls_eps": 0.0,
              "easy_margin": False
              }
    main(CONFIG)
    # main1()