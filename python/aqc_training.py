#! /usr/bin/env python3
# -*- coding: utf-8 -*-

#
# @author Vladimir S. FONOV
# @date 13/04/2018
import argparse
import os
import numpy as np
import io
import copy

from aqc_data   import *
from model.util import *

import torch
import torch.nn as nn

from torch import optim
from torch.utils.tensorboard import SummaryWriter


def parse_options():

    parser = argparse.ArgumentParser(description='Train automated QC',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--n_epochs", type=int, default=10,
                        help="Total number of epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of workers to load data")
    parser.add_argument("--ref",action="store_true",default=False,
                        help="Use reference images")
    parser.add_argument("output", type=str, 
                        help="Output prefix")
    parser.add_argument("--load", type=str, default=None,
                        help="Load pretrained model")
    parser.add_argument("--val",action="store_true",default=False,
                        help="Validate that all files are there") 
    parser.add_argument("--net", choices=['r18', 'r34', 'r50','r101','r152',
                                          'sq101',
                                          'x50', 'x101',
                                          'wr50','wr101'],
                    help="Network type",default='r18')
    parser.add_argument("--adam",action="store_true",default=False,
                        help="Use ADAM instead of SGD") 
    parser.add_argument("--pretrained",action="store_true",default=False,
                        help="Use ImageNet pretrained models") 
    parser.add_argument("--lr",type=float, default=0.0001,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for shuffling data")
    parser.add_argument("--fold", type=int, default=0,
                        help="CV fold")
    parser.add_argument("--folds", type=int, default=0,
                        help="CV total number of folds, 0 - disable CV")

    params = parser.parse_args()
    
    return params

if __name__ == '__main__':
    params = parse_options()
    data_prefix = "../data"
    use_ref = params.ref
    val_subjects = 200
    
    all_samples = load_full_db(data_prefix + os.sep + db_name, data_prefix, False)

    training, validation, testing = split_dataset(all_samples, fold=params.fold, folds=params.folds, validation=val_subjects, 
        shuffle=True, seed=params.seed)

    train_dataset    = QCDataset(training, data_prefix, use_ref=use_ref)
    validate_dataset = QCDataset(validation, data_prefix, use_ref=use_ref)
    testing_dataset = QCDataset(testing, data_prefix, use_ref=use_ref)
    
    print("Training\t{} samples {} unique subjects".format(len(train_dataset),train_dataset.n_subjects()))
    print("Validation\t{} samples {} unique subjects".format(len(validate_dataset),validate_dataset.n_subjects()))
    print("Testing\t{} samples {} unique subjects".format(len(testing_dataset),testing_dataset.n_subjects()))

    dataset_size = len(train_dataset)

    training_dataloader = DataLoader(train_dataset, 
                          batch_size=params.batch_size,
                          shuffle=True, 
                          num_workers=params.workers,
                          drop_last=True,
                          pin_memory=True)
    
    validation_dataloader = DataLoader(validate_dataset, 
                          batch_size=params.batch_size,
                          shuffle=False, 
                          num_workers=params.workers,
                          drop_last=False)

    model = get_qc_model(params, use_ref=use_ref, pretrained=params.pretrained)    


    model     = model.cuda()
    criterion = nn.CrossEntropyLoss()
    if params.adam:
        # parameters from LUA version
        optimizer = optim.Adam(model.parameters(), 
           lr=params.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=params.lr, momentum=0.9, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    writer = SummaryWriter()

    global_ctr = 0
    best_model_acc = copy.deepcopy(model.state_dict())
    best_model_tpr = copy.deepcopy(model.state_dict())
    best_model_tnr = copy.deepcopy(model.state_dict())

    best_acc = 0.0
    best_tnr = 0.0
    best_tpr = 0.0

    validation_period = 200

    for epoch in range(params.n_epochs):
        print('Epoch {}/{}'.format(epoch, params.n_epochs - 1))
        print('-' * 10)

        model.train(True)  # Set model to training mode

        # for stats
        running_loss = 0.0
        running_acc  = 0.0

        val_running_loss = 0.0
        val_running_acc  = 0.0

        val_running_tpr  = 0.0
        val_running_fpr  = 0.0
        val_running_tnr  = 0.0

        val_ctr=0

        for i_batch, sample_batched in enumerate(training_dataloader):
            inputs = sample_batched['image'].cuda()
            labels = sample_batched['status'].cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            # if training
            loss.backward()
            optimizer.step()

            batch_loss = loss.data.item() * inputs.size(0)
            batch_acc  = torch.sum(preds == labels.data).item()

            # statistics
            running_loss  += batch_loss
            running_acc   += batch_acc

            # run validation from time-to time 
            if ( global_ctr % validation_period ) == 0:
                # training stats
                writer.add_scalars('{}/training'.format(params.output),
                                   {'loss': batch_loss/inputs.size(0),
                                    'acc':  batch_acc/inputs.size(0)},
                                    global_ctr)
                # 
                model.train(False)  # Set model to evaluation mode

                v_batch_loss  = 0.0
                v_batch_acc   = 0.0

                v_batch_tp    = 0.0
                v_batch_tn    = 0.0
                v_batch_ap    = 0.0
                v_batch_an    = 0.0
                v_batch_fp    = 0.0

                for v_batch, v_sample_batched in enumerate(validation_dataloader):
                    inputs = v_sample_batched['image' ].cuda()
                    labels = v_sample_batched['status'].cuda()

                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    v_batch_loss += float(loss) * inputs.size(0)
                    v_batch_acc  += float(torch.sum(preds == labels.data))

                    # calculating true positive and true negative
                    v_batch_tp   += float(torch.sum( (preds == 1)*(labels.data==1)))
                    v_batch_fp   += float(torch.sum( (preds == 1)*(labels.data==0)))
                    v_batch_tn   += float(torch.sum( (preds == 0)*(labels.data==0)))

                    v_batch_ap   += float(torch.sum( (labels.data == 1)))
                    v_batch_an   += float(torch.sum( (labels.data == 0)))


                # (?)
                v_batch_loss /= len(validate_dataset)
                v_batch_acc  /= len(validate_dataset)

                if v_batch_ap>0:
                    v_batch_tp  /= v_batch_ap
                else:
                    v_batch_tp = 0.0

                if v_batch_an>0:
                    v_batch_tn  /= v_batch_an
                    v_batch_fp  /= v_batch_an
                else:
                    v_batch_tn = 0.0

                val_running_loss = v_batch_loss
                val_running_acc  = v_batch_acc
                val_running_tnr  = v_batch_tn
                val_running_tpr  = v_batch_tp
                val_running_fpr  = v_batch_fp

                val_ctr += 1
                writer.add_scalars('{}/validation'.format(params.output),
                                   {'loss': v_batch_loss ,
                                    'acc':  v_batch_acc,
                                    'tpr':  v_batch_tp,
                                    'tnr':  v_batch_tn,
                                    'fpr':  v_batch_fp },
                                    global_ctr)

                print("{} - {},{}".format(global_ctr,v_batch_loss,v_batch_acc))
                model.train(True)
            global_ctr += 1
        
        if not params.adam:
            scheduler.step()

        # aggregate epoch statistics           
        epoch_loss = running_loss / dataset_size
        epoch_acc  = running_acc / dataset_size
        
        # TODO: perhaps this is not 
        epoch_val_loss = val_running_loss 
        epoch_val_acc  = val_running_acc 
        epoch_val_tpr  = val_running_tpr
        epoch_val_tnr  = val_running_tnr
        epoch_val_fpr  = val_running_fpr

        writer.add_scalars('{}/validation_epoch'.format(params.output), 
                            {'loss': epoch_val_loss,
                             'acc':  epoch_val_acc,
                             'tpr':  epoch_val_tpr,
                             'tnr':  epoch_val_tnr,
                             'fpr':  epoch_val_fpr
                             },
                            epoch)
        
        print('Epoch: {} Validation Loss: {:.4f} Acc: {:.4f} TPR: {:.4f} TNR: {:.4f} FPR:{:.4f}'.format(epoch, epoch_val_loss, epoch_val_acc, epoch_val_tpr, epoch_val_tnr,epoch_val_fpr))

        if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                best_model_acc = copy.deepcopy(model.state_dict())
                model.train(False)
                save_model(model,"best_acc",params.output)

        if epoch_val_tpr > best_tpr:
                best_tpr = epoch_val_tpr
                best_model_tpr = copy.deepcopy(model.state_dict())
                model.train(False)
                save_model(model,"best_tpr",params.output)

        if epoch_val_tnr > best_tnr:
                best_tnr = epoch_val_tnr
                best_model_tnr = copy.deepcopy(model.state_dict())
                model.train(False)
                save_model(model,"best_tnr",params.output)

        if epoch_val_fpr < best_fpr:
                best_fpr = epoch_val_fpr
                best_model_fpr = copy.deepcopy(model.state_dict())
                model.train(False)
                save_model(model,"best_fpr",params.output)

    model.train(False)
    save_model(model,"final",params.output)

    #writer.export_scalars_to_json(params.output+os.sep+"./all_scalars.json")
    #writer.close()
    
    # get the best models
    if False:
        model.load_state_dict(best_model_acc)
        model=model.cpu()
        save_model(model,"best_acc_cpu",params.output)
        
        model.load_state_dict(best_model_tpr)
        model=model.cpu()
        save_model(model,"best_tpr_cpu",params.output)
        
        model.load_state_dict(best_model_tnr)
        model=model.cpu()
        save_model(model,"best_tnr_cpu",params.output)

        model.load_state_dict(best_model_fpr)
        model=model.cpu()
        save_model(model,"best_fpr_cpu",params.output)
    
