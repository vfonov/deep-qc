import torch
import os

# get models
from .resnet_qc    import *
from .squezenet_qc import *


def load_model(model, to_load):
    """
    load a previously trained model.
    """
    #print('Loading the model from {} ...' .format( to_load))
    model.load_state_dict( torch.load(to_load) )

def save_model(model, name, base, fold=0, folds=0, cpu=False):
    """
    Save the model.
    """
    if not os.path.exists(base):
        os.makedirs(base)
    
    if folds==0:
        path = os.path.join(base, '{}.pth'.format( name) )
    else:
        path = os.path.join(base, '{}_{}_{}.pth'.format( name,fold,folds) )
    
    print('Saving the model to {} ...' .format( path))
    if cpu:
        torch.save(model.cpu().state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


def get_qc_model(params, use_ref=False, pretrained=True, predict_dist=False):
    """
    Generate QC model
     params.net: 'r18', 'r34', 'r50','r101','r152',
                'sq101',
                'x50', 'x101',
                'wr50','wr101'
    """
    num_classes=1 if predict_dist else 2

    if params.net=='r34':
        model=resnet_qc_34(pretrained=(pretrained and params.load is None),use_ref=use_ref,num_classes=num_classes)
    elif params.net=='r50':
        model=resnet_qc_50(pretrained=(pretrained and params.load is None),use_ref=use_ref,num_classes=num_classes)
    elif params.net=='r101':
        model=resnet_qc_101(pretrained=(pretrained and params.load is None),use_ref=use_ref,num_classes=num_classes)
    elif params.net=='r152':
        model=resnet_qc_152(pretrained=(pretrained and params.load is None),use_ref=use_ref,num_classes=num_classes)
    elif params.net=='sq101':
        model=squeezenet_qc(pretrained=(pretrained and params.load is None),use_ref=use_ref,num_classes=num_classes)
    elif params.net=='r18':
        model=resnet_qc_18(pretrained=(pretrained and params.load is None),use_ref=use_ref,num_classes=num_classes)
    elif params.net=='x50': 
        model=resnext_qc_50_32x4d(pretrained=(pretrained and params.load is None),use_ref=use_ref,num_classes=num_classes)
    elif params.net=='x101': 
        model=resnext_qc_101_32x8d(pretrained=(pretrained and params.load is None),use_ref=use_ref,num_classes=num_classes)
    elif params.net=='wr50': 
        model=wide_resnet_qc_50_2(pretrained=(pretrained and params.load is None),use_ref=use_ref,num_classes=num_classes)
    elif params.net=='wr101': 
        model=wide_resnet_qc_101_2(pretrained=(pretrained and params.load is None),use_ref=use_ref,num_classes=num_classes)
    else:
        raise("Unsupported model:"+params.net)
    
    if params.load is not None:
        load_model(model, params.load)

    return model



def clip_grad_norm(parameters, max_norm, norm_type=2):
    """Clips gradient norm of an iterable of parameters.
    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    Arguments:
        parameters (Iterable[Variable]): an iterable of Variables that will have
            gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for infinity norm.
    Returns: grad norm before clipping (for logging mostly)
    """
    parameters = list(parameters)
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.norm(norm_type)
                total_norm += param_norm ** norm_type

        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef >= 1:
        return total_norm
    for p in parameters:
        if p.grad is not None:
            p.grad.mul_(clip_coef)
    
    return total_norm


def get_model_grad_norm(model,norm_type=2):
    parameters = model.parameters()
    parameters = list(parameters)

    if norm_type == float('inf'):
        total_norm = max(p.grad.abs().max() for p in parameters)
    else:
        total_norm = 0.0
        for p in parameters:
            if p.grad is None:
                # should not happen ?
                return None
            param_norm = p.grad.norm(norm_type)
            total_norm += param_norm ** norm_type

    return float(total_norm)

def model_param_norm(model,norm_type=2):
    # based on https://discuss.pytorch.org/t/how-does-one-implement-weight-regularization-l1-or-l2-manually-without-optimum/7951
    parameters = model.parameters()

    total_norm = None
    for p in model.parameters():
        param_norm = p.norm(norm_type)
        if total_norm is not None:
            total_norm += param_norm ** norm_type
        else:
            total_norm = param_norm ** norm_type

    return total_norm


def get_grad_norms(model, norm_type=2):
    """
    Get grad norms
    :param model:  torch.nn.ModuleDict
    :param norm_type: norm type, default L2
    :return: dict with norms
    """
    r={}
    for m in model:
        r.update({m:get_model_grad_norm(model[m])})

    return r
