import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import math, time, copy

import utils, parameters, Unet_models
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
dir_name        = "20220901_bgerr" + str(parameters.background_err)
print(dir_name)

training_set_npz = np.load('dataset/N' + str(parameters.sigNoise) + '_training_set_small.npz')
x_train_obs = training_set_npz['x_train_obs']
x_train = training_set_npz['x_train']
mask_train = training_set_npz['mask_train']

x_val_obs = training_set_npz['x_val_obs']
x_val = training_set_npz['x_val']
mask_val = training_set_npz['mask_val']

stdTr = training_set_npz['std']
meanTr = training_set_npz['mean']

batchsize = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_dataset  = torch.utils.data.TensorDataset(torch.Tensor(x_train_obs), torch.Tensor(x_train), torch.Tensor(mask_train))
val_dataset       = torch.utils.data.TensorDataset(torch.Tensor(x_val_obs),  torch.Tensor(x_val), torch.Tensor(mask_val)) 

dataloaders = {
    'train': torch.utils.data.DataLoader(training_dataset, batch_size=batchsize, shuffle=True, num_workers=4, pin_memory=False),
    'val': torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=True, num_workers=4, pin_memory=False),
}

dataset_sizes = {'train': len(training_dataset), 'val': len(val_dataset)}

model_head = Unet_models.L96_UnetConvRec_head().to(device)
model_head.load_state_dict(torch.load("ckpts/" + dir_name + "/pretrain_head_epoch20"))

model_dyn = Unet_models.L96_UnetConvRec_dyn().to(device)

optimizer_model_dyn = optim.Adam(model_dyn.parameters(), lr=1e-3)

# training function for Generator
since = time.time()

best_model_dyn_wts = copy.deepcopy(model_dyn.state_dict())

best_loss_rec = 1e10

train_loss_rec_list = []
val_loss_rec_list = []
train_loss_dyn_list = []
val_loss_dyn_list = []
train_loss_dynbg_list = []
val_loss_dynbg_list = []
train_loss_R_list = []
val_loss_R_list = []
train_loss_I_list = []
val_loss_I_list = []

num_epochs = 200
model_head.eval()

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model_dyn.train()
        else:
            model_dyn.eval()

        running_loss_rec    = 0.0
        running_loss_dyn    = 0.0
        running_loss_dyn_bg = 0.0
        running_loss_dyn_gt = 0.0
        running_loss_R      = 0.0
        running_loss_I      = 0.0
        num_loss            = 0

        # Iterate over data.
        for inputs, targets, mask, in dataloaders[phase]:
            mask        = mask.to(device)
            targets     = targets.to(device)
            inputs      = inputs.to(device)
            
            optimizer_model_dyn.zero_grad()

            with torch.set_grad_enabled(True): 
                inputs     = model_head(inputs * mask)
                outputs    = model_dyn(inputs)
                
                loss_rec    = torch.mean((outputs - targets)**2)
                loss_dyn_bg = utils.dynamic_loss(outputs, 1, meanTr, stdTr, 3)
                loss_dyn_gt = utils.dynamic_loss(targets, 1, meanTr, stdTr, 3)
                loss_dyn    = utils.dynamic_loss(outputs, 1, meanTr, stdTr, 1)
                loss_R      = torch.sum((outputs - targets)**2 * mask) / torch.sum(mask)
                loss_I      = torch.sum((outputs - targets)**2 * (1 - mask)) / torch.sum(1 - mask)

                loss       = loss_dyn_bg

                if phase == 'train':
                    loss.backward()
                    optimizer_model_dyn.step()

            running_loss_rec         += loss_rec.item()    * inputs.size(0) * stdTr**2
            running_loss_dyn         += loss_dyn.item()    * inputs.size(0) * stdTr**2
            running_loss_dyn_bg      += loss_dyn_bg.item() * inputs.size(0) * stdTr**2
            running_loss_dyn_gt      += loss_dyn_gt.item() * inputs.size(0) * stdTr**2
            running_loss_R           += loss_R.item()      * inputs.size(0) * stdTr**2
            running_loss_I           += loss_I.item()      * inputs.size(0) * stdTr**2
            num_loss                 += inputs.size(0)

        epoch_loss_rec       = running_loss_rec    / num_loss
        epoch_loss_dyn       = running_loss_dyn    / num_loss
        epoch_loss_dyn_bg    = running_loss_dyn_bg / num_loss
        epoch_loss_dyn_gt    = running_loss_dyn_gt / num_loss
        epoch_loss_R         = running_loss_R      / num_loss
        epoch_loss_I         = running_loss_I      / num_loss
        
        if epoch == 0:
            print('dyn loss(gt): {:.4e}'.format(epoch_loss_dyn_gt))
        print('{} rec loss: {:.4e} dyn loss: {:.4e} dyn loss(bg): {:.4e} loss_R: {:.4e} loss_I: {:.4e}'.format(
            phase, epoch_loss_rec, epoch_loss_dyn, epoch_loss_dyn_bg, epoch_loss_R, epoch_loss_I))
        
        if phase == 'train':
            train_loss_rec_list.append(epoch_loss_rec)
            train_loss_dyn_list.append(epoch_loss_dyn)
            train_loss_dynbg_list.append(epoch_loss_dyn_bg)
            train_loss_R_list.append(epoch_loss_R)
            train_loss_I_list.append(epoch_loss_I)
        else:
            val_loss_rec_list.append(epoch_loss_rec)
            val_loss_dyn_list.append(epoch_loss_dyn)
            val_loss_dynbg_list.append(epoch_loss_dyn_bg)
            val_loss_R_list.append(epoch_loss_R)
            val_loss_I_list.append(epoch_loss_I)

        if phase == 'val' and epoch_loss_rec < best_loss_rec:
            best_loss_rec = epoch_loss_rec
            best_model_dyn_wts = copy.deepcopy(model_dyn.state_dict())

    if epoch_loss_dyn_bg < parameters.relative_err:
        break
        
    print()

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('Best val reconstruction loss: {:4e}'.format(best_loss_rec))

save_dir_model_dyn = "ckpts/" + dir_name + "/pretrain_dyn_epoch200"
print("saving model at " + save_dir_model_dyn)
torch.save(best_model_dyn_wts, save_dir_model_dyn)

save_dir_loss_dyn  = "train_loss/" + dir_name + "/pretrain_dyn_epoch" + str(epoch + 1)
print("saving loss at " + save_dir_loss_dyn)
np.savez(save_dir_loss_dyn,
         train_loss_rec   = train_loss_rec_list,   val_loss_rec   = val_loss_rec_list, 
         train_loss_dyn   = train_loss_dyn_list,   val_loss_dyn   = val_loss_dyn_list,
         train_loss_dynbg = train_loss_dynbg_list, val_loss_dynbg = val_loss_dynbg_list,
         train_loss_R     = train_loss_R_list,     val_loss_R     = val_loss_R_list, 
         train_loss_I     = train_loss_I_list,     val_loss_I     = val_loss_I_list,
         time = time_elapsed)