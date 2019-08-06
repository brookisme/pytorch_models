import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_models.helpers import DEFAULT_DROPOUT_RATE
from pytorch_models.helpers import activation, same_padding
#
# BLOCKS
#
class Conv(nn.Module):
    r""" Convolutional Stack

    Each conv block is a convolution optionally followed by:
        1. BatchNorm
        2. Activation
        3. Dropout

    Args:
        in_ch<int>: Number of channels in input
        out_ch<int|None>: 
            - number of channels in output
            - only valid if out_chs is None
            - if None, out_ch=in_ch
        out_chs<list|None>:
            - list of output channels from each conv
            - if None out_chs=[out_ch]*depth
        depth<int>: 
            - the number of convolutional layers 
            - only valid if out_chs is None
        kernel_size<int>: 
            - kernel size
            - only used if kernel_sizes is None
        kernel_sizes<list|None>:
            - the kernel size for each conv
            - if None kernel_sizes=[kernel_size]*len(out_chs)
        stride<int>: 
            - strid
            - only used if strides is None
        strides<int>: 
            - the strides for each conv
            - if None strides=[stride]*len(out_chs)
        dilation<int>: dilation rate
        padding<int|str>: 
            - padding
            - int or 'same' 
        batch_norm<bool>: add batch norm after each conv
        dropout<bool|float>:
            - if truthy dropout applied after each conv
            - if float dropout rate = dropout
            - else dropout rate=0.5
        act<str|func|False>: activation method or method_name
        act_config<dict>: kwarg-dict for activation function after each conv
    """
    #
    # CONSTANTS
    #
    SAME='same'
    RELU='ReLU'


    #
    # PUBLIC METHODS
    #
    def __init__(self,
            in_ch,
            out_ch=None,
            out_chs=None,
            depth=1, 
            kernel_size=3, 
            kernel_sizes=None,
            stride=1,
            strides=None,
            dilation=1,
            padding=SAME, 
            batch_norm=True,
            dropout=False,
            act=RELU,
            act_config={}):
        super(Conv, self).__init__()
        self.in_ch=int(in_ch)
        if out_ch is None:
            out_ch=self.in_ch
        else:
            out_ch=int(out_ch)
        if out_chs is None:
            out_chs=[out_ch]*depth
        self.out_ch=out_chs[-1]
        if kernel_sizes is None:
            kernel_sizes=[kernel_size]*len(out_chs)
        if strides is None:
            strides=[stride]*len(out_chs)
        self.padding=padding
        self.conv_blocks=self._conv_blocks(
            in_ch,
            out_chs,
            kernel_sizes,
            strides,
            dilation,
            batch_norm,
            dropout,
            act,
            act_config)

        
    def forward(self, x):
        return self.conv_blocks(x)


    #
    # INTERNAL METHODS
    #    
    def _conv_blocks(
            self,
            in_ch,
            out_chs,
            kernel_sizes,
            strides,
            dilation,
            batch_norm,
            dropout,
            act,
            act_config):
        layers=[]
        for ch,k,s in zip(out_chs,kernel_sizes,strides):
            layers.append(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=ch,
                    kernel_size=k,
                    stride=s,
                    padding=self._padding(k,dilation),
                    dilation=dilation,
                    bias=(not batch_norm)))
            if batch_norm:
                layers.append(nn.BatchNorm2d(ch))
            if act:
                layers.append(activation(act=act,**act_config))
            if dropout:
                layers.append(nn.Dropout2d(p=dropout))
            in_ch=ch
        return nn.Sequential(*layers)


    def _padding(self,kernel_size,dilation):
        if self.padding==Conv.SAME:
            return same_padding(kernel_size,dilation)
        else:
            return self.padding


class Residual(nn.Module):
    r""" Residual Block

    Args:
        in_ch<int>: 
            - Number of channels in input
            - if None try block.in_ch
        out_ch<int|None>: 
            - number of channels in output
            - if None try block.out_ch
        block<nn.Module|None>:
            - inner block for residual block
            - if None uses Conv through conv_kwargs
        shortcut_method<None|str>
            - method for managing shortcuts with increasing dimensions. options:
            - Residual.IDENTITY_SHORTCUT | None: use identity
            - Residual.ZERO_PADDING_SHORTCUT: add zero padding
            - Residual.CONV_SHORTCUT: use a 1x1 conv to increase the padding
        **conv_kwargs:
            - kwargs for Conv bock (see Conv)
            - ignored if <block> is passed   
    """
    #
    # CONSTANTS
    #
    IDENTITY_SHORTCUT='identity'
    ZERO_PADDING_SHORTCUT='zero_padding'
    CONV_SHORTCUT='conv'


    #
    # PUBLIC METHODS
    #
    def __init__(self,
            in_ch=None,
            out_ch=None,
            block=None,
            shortcut_method=IDENTITY_SHORTCUT,
            **conv_kwargs):
        super(Residual, self).__init__()
        if not block:
            block=Conv(in_ch=in_ch,out_ch=out_ch,**conv_kwargs)
        self.block=block
        if not in_ch:
            in_ch=block.in_ch
        if not in_ch:
            out_ch=block.out_ch
        self._set_shortcut(shortcut_method,in_ch,out_ch)


    def forward(self, x):
        return self.shortcut(self._zpad(x)).add_(self.block(x))


    def _set_shortcut(self,method,in_ch,out_ch):
        if method==Residual.CONV_SHORTCUT:
            self.shortcut=nn.Conv2d(
                in_ch, 
                out_ch, 
                kernel_size=1, 
                stride=1, 
                bias=False)
            self.zero_pad=False
        else:
            self.shortcut=nn.Identity()
            if method==Residual.ZERO_PADDING_SHORTCUT:
                self.zero_pad=out_ch-in_ch
            else:
                self.zero_pad=False


    def _zpad(self,x):
        if self.zero_pad:
            shape=x.shape
            print(shape)
            z=torch.zeros(shape[0],self.zero_pad,shape[2],shape[3])
            print(z.shape)
            x=torch.cat([x,z],dim=1)
        return x



class Dense(nn.Module):
    r""" Dense/Linear Stack

    Each dense block is a dense layer optionally followed by:
        1. BatchNorm
        2. Activation
        3. Dropout

    Args:
        in_ch<int>: Number of features in input
        out_ch<int|None>: 
            - number of features in output
            - only valid if out_chs is None
            - if None, out_ch=in_ch
        out_chs<list|None>:
            - list of output features from each dense/linear layer 
            - if None out_chs=[out_ch]*depth
        depth<int>: 
            - the number of dense/linear layers 
            - only valid if out_chs is None
        batch_norm<bool>: add batch norm after each dense/linear layer 
        dropout<bool|float>:
            - if truthy dropout applied after each dense/linear layer 
            - if float dropout rate = dropout
            - else dropout rate=0.5
        last_act<bool>: 
            - by default activation is ignored for the last dense layer.
            - if true, all dense layers will be followed by the activation
        act<str|func|False>: activation method or method_name
        act_config<dict>: kwarg-dict for activation function after each dense/linear layer 
    """
    #
    # PUBLIC METHODS
    #
    def __init__(self,
            in_ch,
            out_ch=None,
            out_chs=None,
            depth=1, 
            batch_norm=True,
            dropout=False,
            last_act=False,
            act='ReLU',
            act_config={}):
        super(Dense, self).__init__()
        self.in_ch=int(in_ch)
        if out_ch is None:
            out_ch=self.in_ch
        else:
            out_ch=int(out_ch)
        if out_chs is None:
            out_chs=[out_ch]*depth
        if dropout is True:
            dropout=DEFAULT_DROPOUT_RATE
        self.out_ch=out_chs[-1]
        self.dense_blocks=self._dense_blocks(
            self.in_ch,
            out_chs,
            batch_norm,
            dropout,
            last_act,
            act,
            act_config)

        
    def forward(self, x):
        return self.dense_blocks(x)


    #
    # INTERNAL METHODS
    #    
    def _dense_blocks(
            self,
            in_ch,
            out_chs,
            batch_norm,
            dropout,
            last_act,
            act,
            act_config):
        layers=[]
        nb_dense=len(out_chs)
        for i,ch in enumerate(out_chs,start=1):
            layers.append(
                nn.Linear(
                    in_features=in_ch,
                    out_features=ch,
                    bias=(not batch_norm)))
            if batch_norm:
                layers.append(nn.BatchNorm1d(ch))
            if act and (last_act or (i!=nb_dense)):
                layers.append(activation(act=act,**act_config))
            if dropout:
                layers.append(nn.Dropout2d(p=dropout))
            in_ch=ch
        return nn.Sequential(*layers)


