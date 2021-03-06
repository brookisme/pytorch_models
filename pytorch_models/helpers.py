import re
import torch.nn as nn
#
# CONSTANTS
#
SOFTMAX='Softmax'
LOG_SOFTMAX='LogSoftmax'
SIGMOID='Sigmoid'
DEFAULT_DROPOUT_RATE=0.5



#
# MODELING
#
def same_padding(kernel_size,dilation=1):
    """ same-padding """
    size=kernel_size+((kernel_size-1)*(dilation-1))
    return int((size-1)//2)



#
# ACTIVATIONS
#
def activation(act=None,**act_config):
    """ get activation
    Args:
        act<str|func|False|None>: activation method or method name
        act_config<dict>: kwarg-dict for activation method
    Returns:
        instance of activation method
    """
    if isinstance(act,str):
        act=to_camel(act)
        act=re.sub('elu','ELU',act)
        act=re.sub('Elu','ELU',act)
        act=re.sub('rELU','ReLU',act)
        act=re.sub('RELU','ReLU',act)
        act=getattr(nn,act)(**act_config)
    elif act and callable(act()):
        act=act(**act_config)
    return act 


def output_activation(act=False,out_ch=None,multi_label=False,**act_config):
    """ get output_activation

    * similar to `activation` but focused on output activations
    * if (Log)Softmax adds dim=1
    * if act=None, uses Sigmoid or Softmax determined by out_ch

    Args:
        act<str|func|False|None>: activation method or method name
        out_ch<int|None>:
            - number of output channels (classes)
            - only necessary if out_ch is None
        act_config<dict>: kwarg-dict for activation method
    Returns:
        instance of activation method
    """    
    if isinstance(act,str):
        act=to_camel(act)
        if act in [SOFTMAX,LOG_SOFTMAX]:
            act_config['dim']=act_config.get('dim',1)
        act=getattr(nn,act)(**act_config)
    elif act is None:
        if multi_label or out_ch==1:
            act=nn.Sigmoid()
        else:
            act=nn.Softmax(dim=1)
    elif act and callable(act()):
        act=act(**act_config)
    return act 




#
# CONFIG
#
def parse_dropout(dropout):
    if dropout:
        if dropout is True:
            dropout=DEFAULT_DROPOUT_RATE
        else:
            dropout=dropout
        include_dropout=True
    else:
        dropout=0.0
        include_dropout=False 
    return dropout, include_dropout




#
# UTILS
#
def to_camel(string):
    parts=string.split('_')
    return ''.join([p.title() for p in parts])


