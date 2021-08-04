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


def run_validation_testing_loop(dataloader, model, loss_fn=nn.functional.cross_entropy, details=False,predict_dist=False ):
    """
    Run Validation/Testing loop
    return validation_dict, validation_log
    """
    res = {}

    ids     = []
    _preds  = np.zeros(0,dtype='int')
    _labels = np.zeros(0,dtype='int')
    _scores = np.zeros(0)
    _dist   = np.zeros(0)

    val_loss  = 0.0
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import roc_auc_score

    with torch.no_grad():
        for v_batch, v_sample_batched in enumerate(dataloader):
            inputs = v_sample_batched['image' ].cuda()
            labels = v_sample_batched['status'].cuda()

            outputs=model(inputs)
            if predict_dist:
                dist = v_sample_batched['dist'].float().cuda()
                outputs=outputs.squeeze(1)
                loss = loss_fn(outputs, dist)
            else:
                loss = loss_fn(outputs, labels)

            if predict_dist:
                _preds = np.concatenate((_preds, outputs.cpu().numpy()))
                _dist  = np.concatenate((_dist,  dist.cpu().numpy()))
            else:
                outputs = nn.functional.softmax(outputs,1)
                _, preds = torch.max(outputs, 1)
                _preds =np.concatenate((_preds,  preds.cpu().numpy()))
                _scores=np.concatenate((_scores,outputs[:,1].cpu().numpy()))

            _labels=np.concatenate((_labels,labels.cpu().numpy()))
            val_loss += float(loss) * inputs.size(0)
            ids.extend(v_sample_batched['id'])


        # (?)
        val_loss /= len(ids)

        res['summary'] = { 'loss': val_loss }

        if not predict_dist:
            _ap   = float(np.sum( (_labels == 1)))
            _an   = float(np.sum( (_labels == 0)))

            # calculating true positive and true negative
            _tpr = float(np.sum( (_preds == 1)*(_labels==1)))
            _tnr = float(np.sum( (_preds == 0)*(_labels==0)))

            if _ap>0:
                _tpr  /= _ap
            else:
                _tpr = 0.0

            if _an>0:
                _tnr  /= _an
            else:
                _tnr = 0.0

            prec,recall,fbeta,_ = precision_recall_fscore_support(_labels,_preds,average='binary')
            res['summary'].update(
                { 
                    'acc': accuracy_score(_labels, _preds),
                    'prec': prec,
                    'F': fbeta,
                    'recall': recall,
                    'auc': roc_auc_score(_labels,_scores),
                    'tpr': _tpr,
                    'tnr': _tnr
                } )

        
        if details:
            res['details'] = {
                'ids':ids,
                'preds':_preds.tolist(),
                'labels':_labels.tolist(),
                'dist': _dist.tolist()
            }
            if not predict_dist:
                res['details'].update({
                    'scores':_scores.tolist()
                })
    return res



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
    parser.add_argument("--save_final",action="store_true",default=False,
                        help="Save final model") 
    parser.add_argument("--save_best",action="store_true",default=False,
                        help="Save final best models") 
    parser.add_argument("--save_cpu",action="store_true",default=False,
                        help="Save models for CPU inference") 
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
    parser.add_argument("--warmup_lr",type=float, default=1e-9,
                        help="Warmup learning rate")
    parser.add_argument("--warmup_iter",type=int, default=0,
                        help="Warmup iterations")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for shuffling data")
    parser.add_argument("--fold", type=int, default=0,
                        help="CV fold")
    parser.add_argument("--folds", type=int, default=0,
                        help="CV total number of folds, 0 - disable CV")
    parser.add_argument("--validation", type=int, default=200,
                        help="Number of unique subjects used for validation")
    parser.add_argument("--freq", type=int, default=None,
                        help="Perform frequent validations, every N minibatches (for debugging)")
    parser.add_argument("--clip", type=float, default=0.0,
                        help="Apply gradient clipping")
    parser.add_argument("--l2", type=float, default=None,
                        help="Apply l2 regularization")
    parser.add_argument("--balance",action="store_true",default=False,
                        help="Balance validation and testing sample")
    parser.add_argument("--dist",action="store_true",default=False,
                        help="Predict misregistration distance instead of class membership")

    params = parser.parse_args()
    
    return params

if __name__ == '__main__':
    params = parse_options()
    data_prefix = "../data"
    db_name = "qc_db.sqlite3"
    params.ref = params.ref
    grad_norm = params.clip
    regularize_l2 = params.l2
    init_lr = params.lr
    warmup_lr = params.warmup_lr
    warmup_iter = params.warmup_iter
    predict_dist = params.dist
    
    all_samples_main = load_full_db(data_prefix + os.sep + db_name, 
                   data_prefix, True, table="qc_all")

    # if distance training is required 
    all_samples_aug = load_full_db(data_prefix + os.sep + db_name, 
                   data_prefix, True, table="qc_all_aug",
                   use_variant_dist=params.dist )

    print("Main samples: {}".format(len(all_samples_main)))
    print("Aug  samples: {}".format(len(all_samples_aug)))

    training, validation, testing = split_dataset(
        all_samples_main, fold=params.fold, 
        folds=params.folds, 
        validation=params.validation, 
        shuffle=True, seed=params.seed, 
        sec_samples=all_samples_aug )

    train_dataset    = QCDataset(training, data_prefix,   use_ref=params.ref)
    validate_dataset = QCDataset(validation, data_prefix, use_ref=params.ref)
    testing_dataset  = QCDataset(testing, data_prefix,    use_ref=params.ref)
    
    if params.balance:
        validate_dataset.balance()
        testing_dataset.balance()

    print("Training   {} samples, {} unique subjects, balance {}".format(len(train_dataset),   train_dataset.n_subjects(), train_dataset.get_balance()))
    print("Validation {} samples, {} unique subjects, balance {}".format(len(validate_dataset),validate_dataset.n_subjects(), validate_dataset.get_balance()))
    print("Testing    {} samples, {} unique subjects, balance {}".format(len(testing_dataset), testing_dataset.n_subjects(), testing_dataset.get_balance()))

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

    model = get_qc_model(params, use_ref=params.ref, 
                        pretrained=params.pretrained,
                        predict_dist=predict_dist)

    model = model.cuda()
    #criterion = nn.CrossEntropyLoss()
    if params.adam:
        # parameters from LUA version
        optimizer = optim.Adam(model.parameters(), 
           lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=params.lr, momentum=0.9, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    writer = SummaryWriter()

    global_ctr = 0

    best_model_acc  = copy.deepcopy(model.state_dict())
    best_model_tpr  = copy.deepcopy(model.state_dict())
    best_model_tnr  = copy.deepcopy(model.state_dict())
    best_model_auc  = copy.deepcopy(model.state_dict())
    best_model_loss = copy.deepcopy(model.state_dict())

    best_acc = 0.0
    best_acc_epoch = -1
    best_acc_ctr = -1

    best_tnr = 0.0
    best_tnr_epoch = -1
    best_tnr_ctr = -1

    best_tpr = 0.0
    best_tpr_epoch = -1
    best_tpr_ctr = -1

    best_auc = 0.0
    best_auc_epoch = -1
    best_auc_ctr = -1

    best_loss = 1e10
    best_loss_epoch = -1
    best_loss_ctr = -1


    training_log = []
    validation_log = []
    testing_log = []

    for epoch in range(params.n_epochs):
        print('Epoch {}/{}'.format(epoch+1, params.n_epochs))
        print('-' * 10)

        model.train(True)  # Set model to training mode
        for i_batch, sample_batched in enumerate(training_dataloader):
            if epoch==0 and warmup_iter>0:
                if i_batch == 0:
                    for g in optimizer.param_groups :
                        g[ 'lr' ] = warmup_lr
                elif i_batch == warmup_iter:
                    for g in optimizer.param_groups :
                        g[ 'lr' ] = init_lr

            inputs = sample_batched['image'].cuda()
            if predict_dist:
                dist = sample_batched['dist'].float().cuda()
            else:
                labels = sample_batched['status'].cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            if predict_dist:

                outputs=outputs.squeeze(1)
                preds = outputs.data
                loss = nn.functional.mse_loss(outputs, dist)
            else:
                with torch.no_grad():
                    _, preds = torch.max(outputs.data, 1)
                loss = nn.functional.cross_entropy(outputs, labels)

            if regularize_l2>0.0:
                l2_norm = model_param_norm(model , 2)
                loss = loss + l2_norm * regularize_l2

            # if training
            loss.backward()

            if grad_norm > 0.0:
                grad_log = clip_grad_norm(model.parameters(), grad_norm)
            else:
                grad_log = get_model_grad_norm(model.parameters())

            optimizer.step()

            batch_loss = loss.data.item()

            log={'loss': batch_loss,
                 'grad':float(grad_log)}

            if not predict_dist:
                batch_acc  = torch.sum(preds == labels.data).item()
                log['acc'] = batch_acc/inputs.size(0)
            # training stats
            writer.add_scalars('{}/training'.format(params.output),
                                log, global_ctr)
            log['ctr']=global_ctr
            log['epoch']=epoch

            
            if params.freq is not None and \
                  (global_ctr%params.freq)==0 and \
                  len(validation)>0:
                model.train(False)
                val_info = run_validation_testing_loop(validation_dataloader, model, 
                    loss_fn = nn.functional.mse_loss if predict_dist else nn.functional.cross_entropy,
                    predict_dist=predict_dist,
                    details=False)

                val = val_info['summary']
                
                if predict_dist:
                    if val['loss'] < best_loss:
                            best_loss = val['loss']
                            best_loss_epoch = epoch
                            best_loss_ctr = global_ctr
                            best_model_loss = copy.deepcopy(model.state_dict())
                else:
                    if val['acc'] > best_acc:
                            best_acc = val['acc']
                            best_acc_epoch = epoch
                            best_acc_ctr = global_ctr
                            best_model_acc = copy.deepcopy(model.state_dict())

                    if val['tpr'] > best_tpr:
                            best_tpr = val['tpr']
                            best_tpr_epoch = epoch
                            best_tpr_ctr = global_ctr
                            best_model_tpr = copy.deepcopy(model.state_dict())

                    if val['tnr'] > best_tnr:
                            best_tnr = val['tnr']
                            best_tnr_epoch = epoch
                            best_tnr_ctr = global_ctr
                            best_model_tnr = copy.deepcopy(model.state_dict())

                    if val['auc'] > best_auc:
                            best_auc = val['auc']
                            best_auc_epoch = epoch
                            best_auc_ctr = global_ctr
                            best_model_auc = copy.deepcopy(model.state_dict())
                
                writer.add_scalars('{}/validation'.format(params.output), 
                                    val,
                                    global_ctr)

                val['epoch']=epoch
                val['ctr']=global_ctr
                validation_log.append(val)
                model.train(True)

            training_log.append(log)
            global_ctr += 1

        model.train(False)  # Set model to evaluation mode
        # run validation at the end of epoch
        if len(validation)>0:
            val_info = run_validation_testing_loop(validation_dataloader,model,details=False,
                    loss_fn = nn.functional.mse_loss if predict_dist else nn.functional.cross_entropy,
                    predict_dist=predict_dist,
            )
            val = val_info['summary']

            if not params.adam:
                scheduler.step()
            
            if predict_dist:
                if val['loss'] < best_loss:
                        best_loss = val['loss']
                        best_loss_epoch = epoch
                        best_loss_ctr = global_ctr
                        best_model_loss = copy.deepcopy(model.state_dict())
            else:
                if val['acc'] > best_acc:
                        best_acc = val['acc']
                        best_acc_epoch = epoch
                        best_acc_ctr = global_ctr
                        best_model_acc = copy.deepcopy(model.state_dict())

                if val['tpr'] > best_tpr:
                        best_tpr = val['tpr']
                        best_tpr_epoch = epoch
                        best_tpr_ctr = global_ctr
                        best_model_tpr = copy.deepcopy(model.state_dict())

                if val['tnr'] > best_tnr:
                        best_tnr = val['tnr']
                        best_tnr_epoch = epoch
                        best_tnr_ctr = global_ctr
                        best_model_tnr = copy.deepcopy(model.state_dict())

                if val['auc'] > best_auc:
                        best_auc = val['auc']
                        best_auc_epoch = epoch
                        best_auc_ctr = global_ctr
                        best_model_auc = copy.deepcopy(model.state_dict())
            
            writer.add_scalars('{}/validation_epoch'.format(params.output), 
                                val,
                                epoch)

            val['epoch']=epoch
            val['ctr']=global_ctr
            validation_log.append(val)

            if predict_dist:
                print('Epoch: {} Validation Loss: {:.4f}'.\
                    format(epoch, val['loss']))
            else:
                print('Epoch: {} Validation Loss: {:.4f} ACC:{:.4f} TPR:{:.4f} TNR:{:.4f} AUC:{:.4f}'.\
                    format(epoch, val['loss'], val['acc'], val['tpr'], val['tnr'],val['auc']))
        else:
            print('Epoch: {} no validation'.format(epoch))
            
    ###
    final_model = copy.deepcopy(model.state_dict())
    if params.save_final:
        save_model(model,"final", params.output, fold=params.fold, folds=params.folds, cpu=params.save_cpu)

    if len(validation)>0 and params.save_best:
        if predict_dist:
            model.load_state_dict(best_model_loss)
            save_model(model,"best_loss", params.output, fold=params.fold, folds=params.folds, cpu=params.save_cpu)
        else:
            model.load_state_dict(best_model_acc)
            save_model(model,"best_acc", params.output, fold=params.fold, folds=params.folds, cpu=params.save_cpu)
            
            model.load_state_dict(best_model_tpr)
            save_model(model,"best_tpr", params.output, fold=params.fold, folds=params.folds, cpu=params.save_cpu)
                
            model.load_state_dict(best_model_tnr)
            save_model(model,"best_tnr", params.output, fold=params.fold, folds=params.folds, cpu=params.save_cpu)
            
            model.load_state_dict(best_model_auc)
            save_model(model,"best_auc", params.output, fold=params.fold, folds=params.folds, cpu=params.save_cpu)

    testing_final={}
    testing_best_acc={}
    testing_best_tpr={}
    testing_best_tnr={}
    testing_best_auc={}
    testing_best_loss={}

    if len(testing)>0:
        print("Testing...")
        testing_dataloader = DataLoader(testing_dataset, 
                            batch_size=params.batch_size,
                            shuffle=False, 
                            num_workers=params.workers,
                            drop_last=False)

        with torch.no_grad():

            model.load_state_dict(final_model)
            testing_final = run_validation_testing_loop(testing_dataloader, model, details=True,
                        loss_fn = nn.functional.mse_loss if predict_dist else nn.functional.cross_entropy,
                        predict_dist=predict_dist)

            if len(validation)>0:
                if predict_dist:
                    model.load_state_dict(best_model_loss)
                    testing_best_loss = run_validation_testing_loop(testing_dataloader, model, details=True,
                        loss_fn = nn.functional.mse_loss,
                        predict_dist=predict_dist)
                else:
                    model.load_state_dict(best_model_acc)
                    testing_best_acc = run_validation_testing_loop(testing_dataloader, model, details=True)

                    model.load_state_dict(best_model_auc)
                    testing_best_auc = run_validation_testing_loop(testing_dataloader, model, details=True)

                    model.load_state_dict(best_model_tpr)
                    testing_best_tpr = run_validation_testing_loop(testing_dataloader, model, details=True)

                    model.load_state_dict(best_model_tnr)
                    testing_best_tnr = run_validation_testing_loop(testing_dataloader, model, details=True)


    if not os.path.exists(params.output):
        os.makedirs(params.output)

    log_path = os.path.join(params.output, 
                'log_{}_{}.json'.format(params.fold,params.folds))

    
    print("Saving log to {}".format(log_path))
    with open(log_path,'w') as f:
        json.dump(
            {
                'folds': params.folds,
                'fold': params.fold,

                ### DEBUG
                #'training_subj':   list(train_dataset.qc_subjects),
                #'validation_subj': list(validate_dataset.qc_subjects),
                #'testing_subj':    list(testing_dataset.qc_subjects),
                ### DEBUB

                'model':      params.net,
                'model_load': params.load,
                
                'ref':        params.ref,
                'batch_size': params.batch_size,
                'n_epochs':   params.n_epochs,
                'pretrained': params.pretrained,
                'adam':       params.adam,
                'lr':         params.lr,

                'grad_norm' : grad_norm,
                'regularize_l2': regularize_l2,
                'init_lr': init_lr,
                'warmup_lr':warmup_lr,
                'warmup_iter': warmup_iter,              

                'best_acc':best_acc, 
                'best_acc_epoch':best_acc_epoch, 
                'best_acc_ctr':best_acc_ctr, 

                'best_tnr':best_tnr, 
                'best_tnr_epoch':best_tnr_epoch, 
                'best_tnr_ctr':best_tnr_ctr, 

                'best_tpr':best_tpr, 
                'best_tpr_epoch':best_tpr_epoch, 
                'best_tpr_ctr':best_tpr_ctr, 

                'best_auc':best_auc, 
                'best_auc_epoch':best_auc_epoch, 
                'best_auc_ctr':best_auc_ctr, 

                'training':training_log,
                'validation': validation_log,
                
                'testing_final':    testing_final,
                'testing_best_acc': testing_best_acc,
                'testing_best_auc': testing_best_auc,
                'testing_best_tpr': testing_best_tpr,
                'testing_best_tnr': testing_best_tnr,
                'testing_best_loss': testing_best_loss
            }, f  )
