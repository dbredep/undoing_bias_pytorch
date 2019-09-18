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
PATH_TO_FEATURES = './feature/LabelMeSpain.mat'
transformed_datasets = {}
transformed_datasets['train'] = dl.PaperDataset(
    path_to_feature=PATH_TO_FEATURES,
    fold='train')
transformed_datasets['test'] = dl.PaperDataset(
    path_to_feature=PATH_TO_FEATURES,
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


model = models.LinearSVM(5376)
model.cuda()
LR = 1e-2
optimizer = optim.SGD([model.w, model.b], lr=LR)




C1 = 10
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

    optimizer = optim.SGD([model.w, model.b], lr=LR / epoch)

    # set model to train or eval mode based on whether we are in train or
    # val; necessary to get correct predictions given batchnorm
    for phase in ['train', 'test']:
        if phase == 'train':
            model.train(True)
        else:
            model.train(False)
            y_pred = []
            y_label = []

        running_loss = 0.0
        i = 0
        total_done = 0
        acc = 0
        # iterate over all data in train/val dataloader:
        for data in dataloaders[phase]:
            step_count += 1
            #optimizer = optim.SGD([model.w, model.b], lr=LR / step_count)
            i += 1
            inputs, labels = data
            batch_size = inputs.shape[0]
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda()).float()
            outputs = model(inputs)

            #pdb.set_trace()

            # calculate gradient and update parameters in train phase
            optimizer.zero_grad()
            loss = 0.5 * torch.mean(model.w ** 2) + C1 * torch.sum(torch.clamp(1 - outputs.t() * labels, min=0))**2

            #loss = 0.5 * torch.sum(model.w ** 2) - C1 * torch.sum(torch.clamp(outputs.t() * labels, max=1))
            if phase == 'train':
                loss.backward()
                optimizer.step()
            else:
                tt = outputs.t() * labels
                if tt[0].item() > 0:
                    acc += 1
                #pdb.set_trace()
                y_pred.append(outputs[0].item())
                y_label.append(labels[0].item())

            running_loss += loss.data * batch_size
            #print ("training batch loss: ", loss.data)

        epoch_loss = running_loss / dataset_sizes[phase]

        if phase == 'train':
            last_train_loss = epoch_loss
        else:
            with open("results/log_train", 'a') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                if(epoch == 1):
                    logwriter.writerow(["epoch", "train_loss", "test_loss", "test_acc", "test_ap"])
                logwriter.writerow([epoch, last_train_loss, epoch_loss, acc / dataset_sizes[phase], ap(y_label, y_pred)])
            print ("test set accuracy: ", acc / dataset_sizes[phase])
            print ("average_precision_score: ", ap(y_label, y_pred))

        print(phase + ' epoch {}:loss {:.4f} with data size {}'.format(
            epoch, epoch_loss, dataset_sizes[phase]))



    total_done += batch_size
    if(total_done % (100 * batch_size) == 0):
        print("completed " + str(total_done) + " so far in epoch")


time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

