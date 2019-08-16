import math
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
# TOOLS
#
class LowLevelFeatures(object):
    ALL='all'
    HALF='half'


    def __init__(self,
            output_stride=False,
            low_level_output=False,
            drop_array=True):
        self.output_stride=output_stride
        self.half_output_stride=int(float(output_stride)/2)
        self.low_level_output=low_level_output
        self.drop_array=drop_array
        self.reset()


    def increment(self,stride=2):
        self.output_stride_state=self.output_stride_state*stride
        if self.output_stride and (self.output_stride_state>=self.output_stride):
            self.dilation*=stride


    def stride(self,stride):
        if self.dilation==1:
            return stride
        else:
            return 1


    def update_low_level_features(self,x,ch,tag=None):
        if self.low_level_output:
            if isinstance(self.low_level_output,int):
                is_low_level_feature=self.low_level_output==self.output_stride_state
            elif isinstance(self.low_level_output,str):
                if self.low_level_output==LowLevelFeatures.ALL:
                    is_low_level_feature=True
                elif self.low_level_output==LowLevelFeatures.HALF:
                    is_low_level_feature=self.half_output_stride==self.output_stride_state
                else:
                    is_low_level_feature=self.low_level_output==tag
            else:
                state_is_in=self.output_stride_state in self.low_level_output
                if tag:
                    tag_is_in=tag in self.low_level_output
                    is_low_level_feature=state_is_in or tag_is_in
                else:
                    is_low_level_feature=state_is_in
            if is_low_level_feature:
                self.low_level_features.append(x)
                self.low_level_channels.append(ch)


    def out(self):
        if self.drop_array and (len(self.low_level_channels)==1):
            return self.low_level_features[0], self.low_level_channels[0]
        else:
            return self.low_level_features, self.low_level_channels


    def reset(self):
        try:
            del(self.low_level_features)
            del(self.low_level_channels)
        except Exception:
            pass
        self.output_stride_state=1
        self.dilation=1
        self.low_level_features=[]
        self.low_level_channels=[]


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


