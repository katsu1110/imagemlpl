"""
Relatively simple image classification / regression baseline using PyTorch

Small modifications of this single .py file may work for many image classification / regression problems at Kaggle etc.
"""

# !pip install timm
import timm

import os
import sys
import glob
import random
import pathlib
import numpy as np
import pandas as pd

from sklearn import metrics, model_selection
from tqdm import tqdm

from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms as T

# ------------------------------------------------
# config
# ------------------------------------------------
# debug mode
DEBUG = False

# take care for a corrupt images without and ending bit
ImageFile.LOAD_TRUNCATED_IMAGES = True

# data path
INPUT_DIR = "../input/atma11"
OUTPUT_DIR = "./"
IMAGE_FOLDER = "photos"

# device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {DEVICE}')

# task
# TASK = 'multiclass'
# TASK = 'binary'
TASK = 'regression'
LABEL = 'target'
OUT_DIM = 1

# image size
SIZE = 224

# epoch
EPOCHS = 10

# batch size
TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 256

# model
MODEL = 'resnet34d'
IS_PRETRAINED = False

# learning rate
LR = 5e-04
WARMUP_EPO = 1

# num workers
NUM_WORKERS = 4

# train
SA = 3
N_SPLITS = 5
CV = 'StratifiedGroupKFold'
GROUPS = 'art_series_id'
TTA = 10
SEED = 42

# mean and std values of EGB channels for imagene dataset
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

# augmentations
train_trans = T.Compose([
    T.RandomGrayscale(p=0.2),
    T.RandomResizedCrop(SIZE),
    T.RandomVerticalFlip(p=0.25),
    T.RandomHorizontalFlip(p=0.5),
#     T.ColorJitter(
#         brightness=0.3,
#         contrast=0.5,
#         saturation=[0.8, 1.3],
#         hue=[-0.05, 0.05],
#     ),
    T.ToTensor(),
    T.Normalize(IMG_MEAN, IMG_STD)
])
valid_trans = T.Compose([
    T.CenterCrop(SIZE),
    T.ToTensor(),
    T.Normalize(IMG_MEAN, IMG_STD)
])

# ------------------------------------------------
# modify, according to your task 
# ------------------------------------------------
# get images
def get_images(image_dir: str='../inputs/comp/photos'):
    """
    get images
    """
    images = glob.glob(f'{image_dir}/*.jpg')
    return images

# make meta_df
def get_meta_df(train, train_images):
    """
    get meta_df with 'id', 'image path' and 'target'
    """
    train_df = train.copy()
    train_df['images'] = train_images
    return train_df

# get model
def create_model(name=MODEL, pretrained=IS_PRETRAINED):
    """
    get pytorch torchvision model
    https://pytorch.org/vision/stable/models.html#
    """
#     model = torchvision.models.__dict__[name](pretrained=pretrained)
    model = timm.create_model(
        name, pretrained=pretrained,
    )
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features=in_features, out_features=OUT_DIM, bias=True)
    return model

def calculate_metrics(y_true, y_pred) -> dict:
    """
    calculate metric
    """    
    return {
        'rmse': metrics.mean_squared_error(y_true, y_pred) ** .5
    }

def submit(test_predictions: list, sub_file_name: str='submission.csv'):
    """
    save submission file
    """
    # mean predictions by folds
    pred_mean = np.array(test_predictions).mean(axis=0)

    # save file
    pd.DataFrame({
        LABEL: pred_mean
    }).to_csv(os.path.join(OUTPUT_DIR, sub_file_name), index=False)

# ------------------------------------------------
# image data-handling convention
# ------------------------------------------------
# dataset class
class ImageDataset(torch.utils.data.Dataset):
    """
    A general image dataset class
    """

    def __init__(
        self
        , images: list=['../input/images/id01.jpg']
        , targets: np.ndarray=np.zeros(1)
        , task : str='binary'
        , resize=(224, 224)
        , augmentations=None
        ):
        self.images = images
        self.targets = targets
        self.task = task
        self.resize = resize
        self.augmentations = augmentations

    def __len__(self):
        """
        Return the total number of samples in the dataset
        """
        return len(self.images)

    def __getitem__(self, item):
        """
        For a given "item" index, return everything we need to train a given model
        """
        # use PIL to open the image
        image = Image.open(self.images[item])

        # convert image to RGB, we have single channel images
        image = image.convert("RGB")

        # get corresponding targets
        if self.targets is not None:
            targets = self.targets[item]
        else:
            targets = np.ones(1)

        # resize, if needed
        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]),
                resample=Image.BILINEAR
            )

        # if we have albumentation augmentations, add them to the image
        if self.augmentations is not None:
            image = self.augmentations(image)

        # pytorch expects CHW instead of HWC
        image = np.array(image)
#         image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        # return tensors of image and targets 
        image_tensor = torch.tensor(image, dtype=torch.float)
                
        if self.task == "regression":
            target_tensor = torch.tensor(targets, dtype=torch.float)
        else:
            target_tensor = torch.tensor(targets, dtype=torch.long)
            
        return image_tensor, target_tensor
    
# train model
def train(train_loader, model, optimizer, device):
    """
    train for one epoch
    :param train_loader: the pytorch dataloader (training data)
    :param model: pytorch model
    :param optimizer: optimizer (e.g., adam, sgd...)
    :param device: cuda/cpu
    """

    # train mode
    model.train()

    # go over every batch of data in data loader
    for inputs, targets in train_loader:
        
        # move inputs and targets to device
        inputs = inputs.to(device, dtype=torch.float)
        if TASK == 'regression':
            targets = targets.to(device, dtype=torch.float)
        else:
            targets = targets.to(device, dtype=torch.long)

        # zero grad the optimizer
        optimizer.zero_grad()
        
        # forward step
        outputs = model(inputs)

        # calculate loss
        loss = loss_fn(outputs, targets)
        
        # backward step the loss
        loss.backward()

        # step optimizer
        optimizer.step()                    

def predict(data_loader, model, device):
    """
    predict by trained model
    :parm data_loader: the pytorch dataloader
    :param model: pytorch model
    :param device: cuda/cpu
    """

    # evaludation mode
    model.eval()

    # predicts
    predicts = []

    # no_grad context is used
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device, dtype=torch.float)

            # forward step to get prediction
            output = model(inputs)

            # convert outputs to lists
            if TASK == 'regression':
                output = output.detach().cpu().numpy().tolist()
            else:
                output = output.cpu().argmax(dim=1)

            # extend the list
            predicts.extend(output)

    # return final output
    pred = np.array(predicts).reshape(-1)
    return pred

def seed_torch(seed=46):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # get loss
def loss_fn(outputs, targets):
    """
    calculate pytorch loss (modify if necessary)
    """
    if TASK == 'binary':
        criterion = nn.BCEWithLogitloss()
        return criterion(outputs, targets)
    elif TASK == 'multiclass':
        criterion = nn.CrossEntropyLoss()
        return criterion(outputs, targets)
    elif TASK == 'regression':
        criterion = nn.MSELoss()
        return criterion(outputs, targets.view(-1, 1))
    
# add fold number for cross-validation
def get_folds(train_df: pd.DataFrame, targets: np.ndarray, groups=None, method: str='KFold', n_splits: int=5, seed: int=46):
    """
    add folds info for cross-validation
    """
    # method
    if method.lower() == 'kfold':
        kf = model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        cv = kf.split(train_df)

    elif method.lower() == "stratifiedkfold":
        kf = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        cv = kf.split(train_df, targets)
    
    elif method.lower() == "groupkfold":
        kf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        cv = kf.split(train_df, targets, groups)
    
    elif method.lower() == "stratifiedgroupkfold":
        kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        cv = kf.split(train_df, targets, groups)

    # # add folds
    # train_df['fold'] = np.nan
    # for fold, (train_index, test_index) in enumerate(cv):
    #     train_df['fold'].iloc[train_index] = int(fold)
    return cv

def run_fold(
    model
    , train_images
    , train_targets
    , valid_images
    , valid_targets
    , fold_id
    ) -> np.ndarray:
    """
    run train / valid for one fold
    """

    # -------------------------
    # data loader
    # -------------------------
    # train
    train_dataset = ImageDataset(
        train_images
        , train_targets
        , task=TASK
        , resize=(SIZE, SIZE)
        , augmentations=train_trans
        )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS
    )

    # valid
    valid_dataset = ImageDataset(
        valid_images
        , valid_targets
        , task=TASK
        , resize=(SIZE, SIZE)
        , augmentations=valid_trans
        )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=VALID_BATCH_SIZE, num_workers=NUM_WORKERS
    )

    # optimizer
    optimizer = optim.AdamW(params=model.parameters(), lr=LR, weight_decay=LR/100)

    # scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS-WARMUP_EPO)
    
    for epoch in range(1, EPOCHS + 1):
        print(f'start {epoch}')

        # train
        train(train_loader, model, optimizer, DEVICE)

        # predict on valid
        pred = predict(valid_loader, model, DEVICE)
        score = calculate_metrics(valid_targets, pred)

        print(f'EPOCH {epoch}...SCORE: {score}')
        
        score = list(score.values())[0]
        if epoch == 1:
            score_min = score
            
            # save model weight
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'model_{fold_id}.pth'))
        else:
            if score < score_min:
                score_min = score
                
                # save model weight
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'model_{fold_id}.pth'))

def run_folds(
    train_df
    , test_df
    , seed=42
    ):
    """
    run folds
    """
    
    # test dataloader
    if TTA == 0:
        n_times = 1
        test_aug = valid_trans
    else:
        n_times = TTA
        test_aug = train_trans
        
    test_dataset = ImageDataset(
        test_df['images'].values.tolist()
        , None
        , task=TASK
        , resize=(SIZE, SIZE)
        , augmentations=test_aug
        )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=VALID_BATCH_SIZE, drop_last=False, num_workers=NUM_WORKERS
    )
    test_predictions = []
        
    cv = get_folds(train_df, train_df[LABEL], groups=GROUPS, method=CV, n_splits=N_SPLITS, seed=seed)
    for fold_num, (idx_tr, idx_val) in enumerate(cv):
        model = create_model()
        model.to(DEVICE)

        # run fold
        run_fold(
            model
            , train_df['images'].values[idx_tr]
            , train_df[LABEL].values[idx_tr]
            , train_df['images'].values[idx_val]
            , train_df[LABEL].values[idx_val]
            , fold_id=f'seed{seed}_fold{fold_num}'
        )

        # load trained model        
        model = create_model()
        model.to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f'model_seed{seed}_fold{fold_num}.pth')))
        
        # predict on test
        print(f"run #{n_times} times / tta={TTA}")
        predictions = []
        for _ in tqdm(range(n_times)):
            y_pred = predict(test_loader, model, DEVICE)
            predictions.append(y_pred)
        y_pred_i = np.array(predictions).mean(axis=0)
        test_predictions.append(y_pred_i)
        del model

    return test_predictions

if __name__ == "__main__":    
    # meta data and targets
    KEY = 'object_id'
    train_df = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))
    test_df = pd.read_csv(os.path.join(INPUT_DIR, 'test.csv'))

    # train, test images
    train_df['images'] = [os.path.join(INPUT_DIR, IMAGE_FOLDER, f'{_id}.jpg') for _id in train_df[KEY].values.tolist()]
    test_df['images'] = [os.path.join(INPUT_DIR, IMAGE_FOLDER, f'{_id}.jpg') for _id in test_df[KEY].values.tolist()]
    
    if DEBUG:
        train_images = train_images[:10]
        CV = 'KFold'
        EPOCHS = 1
        N_SPLITS = 2
        SA = 2
        TTA = 2
        print('DEBUG MODE')

    # fit and predicts
    test_predictions = []
    for s in range(SA):
        # set seed
        seed = SEED + s ** 2
        seed_torch(seed)
        
        # run folds
        test_predictions_ = run_folds(train_df, test_df, seed)
        test_predictions += test_predictions_
        
    # submit
    submit(test_predictions, 'submission.csv')