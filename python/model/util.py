import torch
import os

# get models
from .resnet_qc    import *
from .squezenet_qc import *


def load_model(model, to_load):
    """
    load a previously trained model.
    """
    print('Loading the model from {} ...' .format( to_load))
    model.load_state_dict( torch.load(to_load) )

def save_model(model, name, base):
    """
    Save the model.
    """
    if not os.path.exists(base):
        os.makedirs(base)
    
    path = os.path.join(base, '{}.pth'.format( name) )
    print('Saving the model to {} ...' .format( path))
    torch.save(model.state_dict(), path)


def get_qc_model(params, use_ref=False, pretrained=True):
    if params.net=='r34':
        model=resnet_qc_34(pretrained=pretrained and params.load is None,use_ref=use_ref)
    elif params.net=='r50':
        model=resnet_qc_50(pretrained=pretrained and params.load is None,use_ref=use_ref)
    elif params.net=='r101':
        model=resnet_qc_101(pretrained=pretrained and params.load is None,use_ref=use_ref)
    elif params.net=='r152':
        model=resnet_qc_152(pretrained=pretrained and params.load is None,use_ref=use_ref)
    elif params.net=='sq101':
        model=squeezenet_qc(pretrained=pretrained and params.load is None,use_ref=use_ref)
    else:
        model=resnet_qc_18(pretrained=pretrained and params.load is None,use_ref=use_ref)
    
    if params.load is not None:
        load_model(model,params.load)

    return model
