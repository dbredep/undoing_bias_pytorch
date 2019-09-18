from __future__ import print_function, division

# pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.metrics import average_precision_score as ap

# image imports
from skimage import io, transform
from PIL import Image

# general imports
import os
import time
from shutil import copyfile
from shutil import rmtree

# data science imports
import pandas as pd
import numpy as np
import csv
import pdb

import dataloader as dl
import models


use_gpu = torch.cuda.is_available()
gpu_count = torch.cuda.device_count()
print("Available GPU count:" + str(gpu_count))

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(2019)

NUM_EPOCHS = 100
BATCH_SIZE = 1

# create train/val dataloaders
PATH_TO_FEATURES = './feature/'
transformed_datasets = {}
transformed_datasets['train'] = dl.AllDataset(
    path_to_feature_folder=PATH_TO_FEATURES,
    fold='train')
transformed_datasets['test'] = dl.AllDataset(
    path_to_feature_folder=PATH_TO_FEATURES,
    fold='test')

dataloaders = {}
dataloaders['train'] = torch.utils.data.DataLoader(
    transformed_datasets['train'],
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8)

dataloaders['test'] = torch.utils.data.DataLoader(
    transformed_datasets['test'],
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=8)

dataset_sizes = {x: len(transformed_datasets[x]) for x in ['train', 'test']}


dataset_num = 4
model = models.UndoingSVM(5376, dataset_num)
model.cuda()
LR = 1e-2

parameters_dict = [model.w, model.b]
for i in range(dataset_num):
    parameters_dict.append(model.delta_w[i])
    parameters_dict.append(model.delta_b[i])
optimizer = optim.SGD(parameters_dict, lr=LR)




C1 = 10
C2 = 2
since = time.time()
start_epoch = 1
num_epochs= 100
best_loss = 999999
best_epoch = -1
last_train_loss = -1
step_count = 0
acc = 0

# iterate over epochs
for epoch in range(start_epoch, num_epochs + 1):
    print('Epoch {}/{}'.format(epoch, num_epochs))
    print('-' * 10)

    optimizer = optim.SGD(parameters_dict, lr=LR / epoch)

    # set model to train or eval mode based on whether we are in train or
    # val; necessary to get correct predictions given batchnorm
    for phase in ['train', 'test']:
        if phase == 'train':
            model.train(True)
        else:
            model.train(False)
            y_pred1, y_pred2, y_label = [], [], []

        running_loss = 0.0
        i = 0
        total_done = 0
        acc1 = 0
        acc2 = 0
        # iterate over all data in train/val dataloader:
        for data in dataloaders[phase]:
            step_count += 1
            #optimizer = optim.SGD([model.w, model.b], lr=LR / step_count)
            i += 1
            inputs, labels, dataset_ids = data
            batch_size = inputs.shape[0]
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda()).float()
            out_vw, out_bias = model(inputs, dataset_ids)

            d = int(dataset_ids.item())

            # calculate gradient and update parameters in train phase
            optimizer.zero_grad()
            loss_vw = 0.5 * torch.mean(model.w ** 2) + C1 * torch.sum(torch.clamp(1 - out_vw.t() * labels, min=0))**2
            loss_bias = 0.5 * torch.mean(model.delta_w[dataset_ids.item()] ** 2) + C2 * torch.sum(torch.clamp(1 - out_bias.t() * labels, min=0))**2
            loss = loss_vw + loss_bias

            #loss = 0.5 * torch.sum(model.w ** 2) - C1 * torch.sum(torch.clamp(outputs.t() * labels, max=1))
            if phase == 'train':
                loss.backward()
                optimizer.step()
            else:
                t1 = out_vw.t() * labels
                t2 = out_bias.t() * labels
                if t1[0].item() > 0:
                    acc1 += 1
                if t2[0].item() > 0:
                    acc2 += 1
                #pdb.set_trace()
                y_pred1.append(out_vw[0].item())
                y_label.append(labels[0].item())
                y_pred2.append(out_bias[0].item())

            running_loss += loss.data * batch_size
            #print ("training batch loss: ", loss.data)

        epoch_loss = running_loss / dataset_sizes[phase]

        if phase == 'train':
            last_train_loss = epoch_loss
        else:
            with open("results/log_train", 'a') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                if(epoch == 1):
                    logwriter.writerow(["epoch", "train_loss", "test_loss", "test_acc_vw", "test_acc_bias", "test_ap_vw", "test_ap_bias"])
                logwriter.writerow([epoch, last_train_loss, epoch_loss, acc1/dataset_sizes[phase], acc2/dataset_sizes[phase], ap(y_label, y_pred1), ap(y_label, y_pred2)])
            print ("test set accuracy visual world: ", acc1 / dataset_sizes[phase])
            print ("test set accuracy bias dataset: ", acc2 / dataset_sizes[phase])
            print ("average_precision_score visual world: ", ap(y_label, y_pred1))
            print ("average_precision_score bias dataset: ", ap(y_label, y_pred2))

        print(phase + ' epoch {}:loss {:.4f} with data size {}'.format(
            epoch, epoch_loss, dataset_sizes[phase]))

    total_done += batch_size
    if(total_done % (100 * batch_size) == 0):
        print("completed " + str(total_done) + " so far in epoch")


time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

