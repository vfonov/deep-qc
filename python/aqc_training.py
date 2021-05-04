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
import json

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
    db_name = "qc_db.sqlite3"
    use_ref = params.ref
    val_subjects = 200
    
    all_samples = load_full_db(data_prefix + os.sep + db_name, data_prefix, False)
    print("All samples: {}".format(len(all_samples)))

    training, validation, testing = split_dataset(all_samples, fold=params.fold, folds=params.folds, validation=val_subjects, 
        shuffle=True, seed=params.seed)

    print("Training\t {} ".format(len(training)))
    print("Validation\t{} ".format(len(validation)))
    print("Testing\t{} ".format(len(testing)))

    train_dataset    = QCDataset(training, data_prefix, use_ref=use_ref)
    validate_dataset = QCDataset(validation, data_prefix, use_ref=use_ref)
    testing_dataset  = QCDataset(testing, data_prefix, use_ref=use_ref)
    
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
    #criterion = nn.CrossEntropyLoss()
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

    training_log = []
    validation_log = []
    testing_log = []
    testing_details = []

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

        model.train(True)
        for i_batch, sample_batched in enumerate(training_dataloader):
            inputs = sample_batched['image'].cuda()
            labels = sample_batched['status'].cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            with torch.no_grad():
                _, preds = torch.max(outputs.data, 1)
            loss = nn.functional.cross_entropy(outputs, labels)

            # if training
            loss.backward()
            optimizer.step()

            batch_loss = loss.data.item() * inputs.size(0)
            batch_acc  = torch.sum(preds == labels.data).item()

            log={'loss': batch_loss/inputs.size(0),
                 'acc':  batch_acc/inputs.size(0)}
            # training stats
            writer.add_scalars('{}/training'.format(params.output),
                                log, global_ctr)
            log['ctr']=global_ctr
            training_log.append(log)
            global_ctr += 1

        model.train(False)  # Set model to evaluation mode
        # run validation at the end of epoch
        with torch.no_grad():
            val_loss  = 0.0
            val_acc   = 0.0

            val_tp    = 0.0
            val_tn    = 0.0
            val_ap    = 0.0
            val_an    = 0.0
            val_fp    = 0.0

            for v_batch, v_sample_batched in enumerate(validation_dataloader):
                inputs = v_sample_batched['image' ].cuda()
                labels = v_sample_batched['status'].cuda()

                outputs=model(inputs)
                loss = nn.functional.cross_entropy(outputs, labels)
                outputs = nn.functional.softmax(outputs,1)
                _, preds = torch.max(outputs, 1)

                val_loss += float(loss) * inputs.size(0)
                val_acc  += float(torch.sum(preds == labels))

                # calculating true positive and true negative
                val_tp   += float(torch.sum( (preds == 1)*(labels==1)))
                val_fp   += float(torch.sum( (preds == 1)*(labels==0)))
                val_tn   += float(torch.sum( (preds == 0)*(labels==0)))

                val_ap   += float(torch.sum( (labels == 1)))
                val_an   += float(torch.sum( (labels == 0)))

            # (?)
            val_loss /= len(validate_dataset)
            val_acc  /= len(validate_dataset)

            if val_ap>0:
                val_tp  /= val_ap
            else:
                val_tp = 0.0

            if val_an>0:
                val_tn  /= val_an
                val_fp  /= val_an
            else:
                val_tn = 0.0
            
            print("{} - {},{}".format(global_ctr, val_loss, val_acc))

            log = {'loss': val_loss,
                                'acc':  val_acc,
                                'tpr':  val_tp,
                                'tnr':  val_tn,
                                'fpr':  val_fp
                                }

            writer.add_scalars('{}/validation_epoch'.format(params.output), 
                                log,
                                epoch)

            log['ctr']   = global_ctr
            log['epoch'] = epoch
            validation_log.append(log)

            print('Epoch: {} Validation Loss: {:.4f} Acc: {:.4f} TPR: {:.4f} TNR: {:.4f} FPR:{:.4f}'.\
                    format(epoch, val_loss, val_acc, val_tp, val_tn,val_fp))
                
        if not params.adam:
            scheduler.step()
        
        if val_acc > best_acc:
                best_acc = val_acc
                best_model_acc = copy.deepcopy(model.state_dict())
                save_model(model,"best_acc",params.output,fold=params.fold,folds=params.folds)

        if val_tp > best_tpr:
                best_tpr = val_tp
                best_model_tpr = copy.deepcopy(model.state_dict())
                save_model(model,"best_tpr",params.output,fold=params.fold,folds=params.folds)

        if val_tn > best_tnr:
                best_tnr = val_tn
                best_model_tnr = copy.deepcopy(model.state_dict())
                save_model(model,"best_tnr",params.output,fold=params.fold,folds=params.folds)

    save_model(model,"final", params.output,fold=params.fold,folds=params.folds)
    
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
    
    if len(testing)>0:
        testing_dataloader = DataLoader(testing_dataset, 
                            batch_size=1,
                            shuffle=False, 
                            num_workers=params.workers,
                            drop_last=False)

        with torch.no_grad():
            val_loss  = 0.0
            val_acc   = 0.0

            val_tp    = 0.0
            val_tn    = 0.0
            val_ap    = 0.0
            val_an    = 0.0
            val_fp    = 0.0

            for v_batch, v_sample_batched in enumerate(testing_dataloader):
                inputs = v_sample_batched['image' ].cuda()
                labels = v_sample_batched['status'].cuda()

                outputs = model(inputs)
                loss = nn.functional.cross_entropy(outputs, labels)
                outputs = nn.functional.softmax(outputs,1)
                _, preds = torch.max(outputs, 1)

                testing_details.append(
                    {
                        'id':    v_sample_batched['id'][0],
                        'label': int(labels[0]),
                        'raw' :  float(outputs[0,1]),
                        'pred':  int(preds[0])
                    }
                )

                val_loss += float(loss) * inputs.size(0)
                val_acc  += float(torch.sum(preds == labels))

                # calculating true positive and true negative
                val_tp   += float(torch.sum( (preds == 1)*(labels==1)))
                val_fp   += float(torch.sum( (preds == 1)*(labels==0)))
                val_tn   += float(torch.sum( (preds == 0)*(labels==0)))

                val_ap   += float(torch.sum( (labels == 1)))
                val_an   += float(torch.sum( (labels == 0)))

            # (?)
            val_loss /= len(testing_dataset)
            val_acc  /= len(testing_dataset)

            if val_ap>0:
                val_tp  /= val_ap
            else:
                val_tp = 0.0

            if val_an>0:
                val_tn  /= val_an
                val_fp  /= val_an
            else:
                val_tn = 0.0
            
            print("{} - {},{}".format(global_ctr, val_loss, val_acc))

            log = {'loss':  val_loss,
                    'acc':  val_acc,
                    'tpr':  val_tp,
                    'tnr':  val_tn,
                    'fpr':  val_fp
                    }

            testing_log.append(log)

            print('Testing Loss: {:.4f} Acc: {:.4f} TPR: {:.4f} TNR: {:.4f} FPR:{:.4f}'.\
                    format(val_loss, val_acc, val_tp, val_tn, val_fp))

    log_path = os.path.join(params.output, 'log_{}_{}.json'.format(params.fold,params.folds))
    print("Saving log to {}".format(log_path))
    with open(log_path,'w') as f:
        json.dump(
            {
                'folds': params.folds,
                'fold': params.fold,
                'model': params.net,
                'ref': params.ref,
                'batch_size': params.batch_size,
                'n_epochs': params.n_epochs,
                'training':training_log,
                'validation': validation_log,
                'testing': testing_log,
                'testing_details': testing_details
            }, f  )
        


    