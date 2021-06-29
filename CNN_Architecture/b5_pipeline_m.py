## Imports #####
import numpy as np
import pandas as pd
import os
import cv2,glob,time, random
import warnings

import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm


import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
from torch.optim import Adam

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
###################################################################
#Configuration

MODEL_ARCH= "efficientnet_b5"
EPOCHS = 15
IMG_SIZE = 512
BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
ITER_FREQ = 200
NUM_WORKERS = 8
SEED = 42
MAX_NORM = 1000
ITERS_TO_ACCUMULATE = 1
SCHEDULER_UPDATE ='epoch' #Can be on a 'batch' basis as well
ITER_VISUALS = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEBUG = False #

def prepare_labels(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    y = onehot_encoded
    return y, label_encoder

class retinopathy2015(Dataset):
    def __init__(self, df, transforms=None):
        #         self.df = df
        self.imageList = df['image'].values
        self.transform = None
        if transforms is None:
            self.transform = A.Compose([
                A.Resize(IMG_SIZE, IMG_SIZE),
                ToTensorV2()
            ])
        else:
            self.transform = transforms
        self.labels, le = prepare_labels(df['level'])

    def __len__(self):
        return len(self.imageList)

    def __getitem__(self, idx):
        file_name = self.imageList[idx]
        img = cv2.imread(f"../resized_train_15/{file_name}.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = self.transform(image=img)
        image = image['image']
        label = self.labels[idx]
        return image, label


class Model(nn.Module):  # EFFNET
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity()
        self.model.classifier = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, num_classes)

    def forward(self, x):
        bs = x.size(0)  # bs -> batch size
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)
        return output


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch(SEED)


def macro_multilabel_auc(label, pred):
    aucs = []
    target_cols = [0, 1, 2, 3, 4]
    for i in range(len(target_cols)):
        aucs.append(roc_auc_score(label[:, i], pred[:, i]))
    return np.mean(aucs)


def train_fn(model, dataloader, device, epoch, optimizer, criterion, scheduler):
    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    model.train()
    scaler = GradScaler()
    start_time = time.time()
    PREDS = []
    TARGETS = []

    loader = tqdm(dataloader, total=len(dataloader))
    for step, (images, labels) in enumerate(loader):

        images = images.to(device).float()
        labels = labels.to(device)
        data_time.update(time.time() - start_time)

        with autocast():
            output = model(images)
            loss = criterion(output, labels)
            #########################
            PREDS += [output.sigmoid()]
            TARGETS += [labels.detach().cpu()]
            ##########################
            losses.update(loss.item(), BATCH_SIZE)
            scaler.scale(loss).backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_NORM)
            if (step + 1) % ITERS_TO_ACCUMULATE == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # if step % ITER_VISUALS == 0 and step != 0:
        #     PRE = torch.cat(PREDS).detach().cpu()
        #     PRE = PRE.numpy()
        #
        #     TAR = torch.cat(TARGETS).numpy()
        #
        #     train_roc_auc = macro_multilabel_auc(TAR, PRE)
        #     t_roc_auc = round(train_roc_auc,2)
        #     train_loss_avg = losses.avg
        #     ep = step / len(dataloader) + epoch
        #     ep = round(ep,2)
        #     val_roc_auc, val_loss_avg = valid_fn(ep,
        #                                          model,
        #                                          val_criterion,
        #                                          val_loader,
        #                                          device,
        #                                          scheduler)
        #
        #     train_roc.append(train_roc_auc)
        #     train_loss.append(train_loss_avg)
        #
        #     val_roc.append(val_roc_auc)
        #     val_loss.append(val_loss_avg)

        if scheduler is not None and SCHEDULER_UPDATE == 'batch':
            scheduler.step()

        batch_time.update(time.time() - start_time)
        start_time = time.time()

        if step % ITER_FREQ == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                  'Data Time {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})'.format((epoch + 1),
                                                 step, len(dataloader),
                                                 batch_time=batch_time,
                                                 data_time=data_time,
                                                 loss=losses
                                                 ))
            # accuracy=accuracies))
        # To check the loss real-time while iterating over data.   'Accuracy {accuracy.val:.4f} ({accuracy.avg:.4f})'
        loader.set_description(f'Training Epoch {epoch + 1}/{EPOCHS}')
        loader.set_postfix(loss=losses.avg)  # accuracy=accuracies.avg)
    #         del images, labels
    #     if scheduler is not None and SCHEDULER_UPDATE == 'epoch':
    #         scheduler.step(losses.avg)

    return losses.avg #, train_roc, train_loss, val_roc, val_loss


def valid_fn(epoch, model, criterion, val_loader, device, scheduler):
    model.eval()
    losses = AverageMeter()
    # accuracies = AverageMeter()
    PREDS = []
    TARGETS = []
    loader = tqdm(val_loader, total=len(val_loader))
    with torch.no_grad():  # without torch.no_grad() will make the CUDA run OOM.
        for step, (images, labels) in enumerate(loader):
            images = images.float().to(device)
            labels = labels.to(device)

            output = model(images)
            loss = criterion(output, labels)
            losses.update(loss.item(), VAL_BATCH_SIZE)
            PREDS += [output.sigmoid()]
            TARGETS += [labels.detach().cpu()]
            loader.set_description(f'Validating Epoch {epoch + 1}/{EPOCHS}')
            loader.set_postfix(loss=losses.avg)  # , accuracy=accuracies.avg)
    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()
    roc_auc = macro_multilabel_auc(TARGETS, PREDS)
    if scheduler is not None and SCHEDULER_UPDATE == 'epoch':
        scheduler.step(losses.avg)
    print(f"Validation ROC AUC -> {round(roc_auc,2)}")
    return losses.avg, roc_auc


def engine(device,df,fold):
    df_train = df[df['folds'] != fold]
    df_val = df[df['folds'] == fold]
    train_data = retinopathy2015(df_train, transforms=None)
    val_data = retinopathy2015(df_val, transforms=None)

    train_loader = DataLoader(train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_WORKERS,
                              pin_memory=True,  # enables faster data transfer to CUDA-enabled GPUs.
                              drop_last=True)
    val_loader = DataLoader(val_data,
                            batch_size=VAL_BATCH_SIZE,
                            num_workers=NUM_WORKERS,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False)

    model = Model('efficientnet_b5', num_classes=5, pretrained=True)

    model.to(device)

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)

    criterion = nn.BCEWithLogitsLoss().to(device)
    val_criterion = nn.BCEWithLogitsLoss().to(device)

    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.8, patience=2)
    loss = []
    accuracy = []

    out_dict = {
        'train_loss': [],
        'val_roc': [],
        'val_loss': []
    }

    START_EPOCH = 0
    for epoch in range(START_EPOCH, EPOCHS):
        epoch_start = time.time()
        avg_loss = train_fn(model, train_loader, device, epoch, optimizer, criterion, scheduler,
                                                      val_criterion,val_loader)
        out_dict['train_loss'].append(avg_loss)
        torch.cuda.empty_cache()
        avg_val_loss, roc_auc_score = valid_fn(epoch, model, val_criterion, val_loader, device, scheduler)
        epoch_end = time.time() - epoch_start

        print(f'Validation accuracy after epoch {epoch + 1}: {roc_auc_score:.4f}')
        loss.append(avg_loss)

        out_dict['val_loss'].append(avg_val_loss)
        out_dict['val_roc'].append(roc_auc_score)

        content = f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f} roc_auc_score: {roc_auc_score:.4f} time: {epoch_end:.0f}s'
        with open(f'GPU_{MODEL_ARCH}.txt', 'a') as appender:
            appender.write(content + '\n')  # avg_train_accuracy: {avg_accuracy:.4f}

        with open(f'{MODEL_ARCH}_ROC_LOSS.json',"w") as out:
            json.dump(out_dict,out)

        torch.save(model.state_dict(), f'{MODEL_ARCH}_epoch_{(epoch + 1)}.pth')
        torch.cuda.empty_cache()

    return loss

if __name__ == '__main__':
    df_2015 = pd.read_csv("../labels/trainLabels15.csv")
    df_2015['level'].value_counts()
    if DEBUG:
        df = df_2015.sample(200).reset_index(drop=True)
    else:
        df = df_2015.sample(frac=1.0, random_state=10).reset_index(drop=True)
    skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
    df['folds'] = 0
    target = df.loc[:, 'level']
    fold_no = 0
    for train_index, test_index in skf.split(df, target):
        df.loc[test_index, 'folds'] = fold_no
        fold_no += 1
    fold = 2
    engine(device, df, fold)


