import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_models.helpers import DEFAULT_DROPOUT_RATE, parse_dropout
from pytorch_models.helpers import activation, same_padding
#
# CONSTANTS
#
CROP_TODO="TODO: Need to crop 1x1 Conv to implement non-same padding"


#
# GENERAL BLOCKS
#
class SeparableConv2d(nn.Module):
    """ stack of SeparableConv2d
    Args:
        in_ch<int>: number of input channels
        out_ch<int|None>: if out_ch is None out_ch=in_ch
        kernel_size<int>: kernel_size
        stride<int>: stride
        dilation<int>: spacing between kernel elements
        padding<int|str>: TODO: for now use default ('same')
        batch_norm<bool>: include batch_norm
        bias<bool|None>: 
            - include bias term.  
            - if None bias=(not batch_norm)
        act<str|func>: activation function after block 
        act_config: kwarg dict for activation function after block
        dropout<float|bool|None>: 
            - after forward pass
            - if is True dropout=0.5
        pointwise_in<bool>: 
            - if true perform 1x1 convolution first (inception)
            - otherwise perform the 1x1 convolution last.
    """ 
    #
    # CONSTANTS
    #
    SAME='same'


    #
    # INSTANCE METHODS
    #
    def __init__(
            self,
            in_ch,
            out_ch=None,
            kernel_size=3,
            stride=1,
            dilation=1,
            padding=SAME,
            batch_norm=True,
            bias=None,
            act='ReLU',
            act_config={},
            dropout=False,
            pointwise_in=True):
        super(SeparableConv2d, self).__init__()
        if not bias:
            bias=(not batch_norm)
        self.pointwise_in=pointwise_in
        if not out_ch:
            out_ch=in_ch
        if self.pointwise_in:
            conv_ch=out_ch
        else:
            conv_ch=in_spadch
        spad=same_padding(kernel_size,dilation)
        if padding==SeparableConv2d.SAME:
            padding=spad
        if padding!=spad:
            raise NotImplementedError(CROP_TODO)
        self.conv=nn.Conv2d(
            in_channels=conv_ch, 
            out_channels=conv_ch, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=conv_ch,
            dilation=dilation,
            bias=bias)
        self.pointwise=nn.Conv2d(
            in_channels=in_ch, 
            out_channels=out_ch, 
            kernel_size=1, 
            stride=1,
            padding=0,
            bias=True )
        self.batch_norm=self._batch_norm(out_ch,batch_norm)
        self.act=activation(act,**act_config)
        self.dropout, self.include_dropout=parse_dropout(dropout)


    def forward(self, x):
        if self.pointwise_in:
            x=self.pointwise(x)
            x=self.conv(x)
        else:
            x=self.conv(x)
            x=self.pointwise(x)
        if self.batch_norm:
            x=self.batch_norm(x)
        if self.act:
            x=self.act(x)
        return F.dropout(
            x,
            p=self.dropout,
            training=self.include_dropout and self.training)


    #
    # INTERNAL
    #
    def _batch_norm(self,out_ch,batch_norm):
        if batch_norm:
            batch_norm=nn.BatchNorm2d(out_ch)
        else:
            batch_norm=False
        return batch_norm


    def _dropout(self,dropout):
        if dropout:
            if dropout is True:
                dropout=DEFAULT_DROPOUT_RATE
            else:
                dropout=dropout
            include_dropout=True
        else:
            dropout=False
            include_dropout=False 
        return dropout, include_dropout



class SeparableStack(nn.Module):
    """ stack of SeparableConv2d
    Args:
        in_ch<int>: number of input channels
        out_chs<list<int>|None>: list of output channels for each SeparableConv2d
        out_ch<int>/depth<int>:
            - use if out_chs is None
            - if out_ch is None out_ch=in_ch
            - create 'depth' number of layers (ie out_chs=[out_ch]*depth)
        stride<int>: stride
        dilation<int>: spacing between kernel elements
        res<bool>:
            - if true add resnet 'ident' to output of SeparableConv2d-Blocks.
            - if in_ch != out_ch perform 1x1 Conv to match channels
        batch_norm/act/act_config/dropout: see SeparableConv2d
    """
    def __init__(self,
            in_ch,
            out_chs=None,
            out_ch=None,
            stride=1,
            dilation=1,
            depth=1,
            res=False,
            batch_norm=True,
            act='ReLU',
            act_config={},
            dropout=False):
        super(SeparableStack, self).__init__()
        if not out_chs:
            if not out_ch:
                out_ch=in_ch
            out_chs=[out_ch]*depth
        self.sconvs=self._sconv_blocks(
            in_ch,
            out_chs,
            stride,
            dilation,
            batch_norm,
            act,
            act_config,
            dropout)
        self.res=res
        self.ident_conv=self._ident_conv(in_ch,out_chs)


    def forward(self,x):
        xout=self.sconvs(x)
        if self.res:
            if self.ident_conv:
                x=self.ident_conv(x)
            return x.add(xout)
        else:
            return xout


    #
    # INTERNAL
    #
    def _sconv_blocks(self,
            in_ch,
            out_chs,
            stride,
            dilation,
            batch_norm,
            act,
            act_config,
            dropout):
        blocks=[]
        for ch in out_chs:
            blocks.append(SeparableConv2d(
                in_ch,
                ch,
                stride=stride,
                dilation=dilation,
                batch_norm=batch_norm,
                act=act,
                act_config=act_config,
                dropout=dropout))
            in_ch=ch
        return nn.Sequential(*blocks)


    def _ident_conv(self,in_ch,out_chs):
        if self.res and in_ch!=out_chs[-1]:
            return nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=1)
        else:
            return False




#
# XCEPTION ARCHITECTURE
#
class EntryBlock(nn.Module):
    """ Xception Entry Block:

    The first two layers of Xception network, before
    any SeparableConv2d blocks

    Args:
        in_ch<int>: number of input channels
        entry_ch<int>: out_ch of first block (the stride-2 block)
        entry_out_ch<int>: out_ch of second block
        act<str|func>: activation function after each conv 
        act_config: kwarg dict for activation function after each conv
    """
    def __init__(
            self,
            in_ch,
            entry_ch=32,
            entry_out_ch=64,
            batch_norm=True,
            bias=None,
            act='ReLU',
            act_config={}
        ):
        super(EntryBlock, self).__init__()
        if bias is None:
            bias=(not batch_norm)
        if batch_norm:
            self.bn1=nn.BatchNorm2d(entry_ch)
            self.bn2=nn.BatchNorm2d(entry_out_ch)
        else:
            self.bn1=False
            self.bn2=False
        self.act1=activation(act,**act_config)
        self.act2=activation(act,**act_config)
        self.conv1=nn.Conv2d(
            in_channels=in_ch,
            out_channels=entry_ch,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias)
        self.conv2=nn.Conv2d(
            in_channels=entry_ch,
            out_channels=entry_out_ch,
            kernel_size=3,
            padding=1,
            bias=bias)
        self.out_ch=entry_out_ch
        self.output_stride=2


    def forward(self,x):
        x=self.conv1(x)
        if self.bn1:
            x=self.bn1(x)
        if self.act1:
            x=self.act1(x)
        x=self.conv2(x)
        if self.bn2:
            x=self.bn2(x)
        if self.act2:
            x=self.act2(x)
        return x




class XBlock(nn.Module):
    """ Xception Block:

    ResStack of SeparableConv2d blocks where the last 3x3 block
    is  SeparableConv2d-stride-2 (for strided=True) or MaxPooling
    for (strided=False)

    the 3x3 stack then looks like:
        * SeparableConv2d(in_ch,out_ch)
        * (depth-2) x SeparableConv2d(out_ch,out_ch)
        * SeparableConv2d(out_ch,out_ch,stride=2) or MaxPooling
    Args:
        in_ch<int>: number of input channels
        out_ch<int>: 
            - number of output channels
            - if out_ch is None, out_ch = in_ch
        depth<int>: 
            - must be >= 2
            - total number of layers
        out_dilation: dilation for the last layer
        maxpool<bool>:
            - if false use SeparableConv2d-stride-(dilation) as the last layer
            - otherwise use MaxPooling as the last layer
        dropout<float|bool|None>: 
            - after forward pass
            - if is True dropout=0.5
    """
    def __init__(self,
            in_ch,
            out_ch=None,
            depth=3,
            dilation=1,
            maxpool=False,
            dropout=False):
        super(XBlock, self).__init__()
        if out_ch is None:
            out_ch=in_ch
        self.sconv_in=SeparableConv2d(
            in_ch,
            out_ch,
            dilation=dilation)
        self.sconv_blocks_depth=depth-2
        if self.sconv_blocks_depth:
            self.sconv_blocks=SeparableStack(
                    out_ch,
                    depth=self.sconv_blocks_depth,
                    dilation=dilation)
        if dilation==1:
            self.output_stride=2
        else:
            self.output_stride=1
        self.reduction_layer=self._reduction_layer(
            out_ch,
            maxpool,
            dilation)
        self.pointwise_conv=nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            stride=self.output_stride,
            kernel_size=1)
        self.pointwise_bn=nn.BatchNorm2d(out_ch)
        self.dropout, self.include_dropout=parse_dropout(dropout)
        self.out_ch=out_ch
        

    def forward(self,x):
        xpc=self.pointwise_conv(x)
        xpc=self.pointwise_bn(xpc)
        x=self.sconv_in(x)
        if self.sconv_blocks_depth:
            x=self.sconv_blocks(x)
        x=self.reduction_layer(x)
        x=xpc.add(x)
        return F.dropout(
            x,
            p=self.dropout,
            training=self.include_dropout and self.training)


    #
    # INTERNAL
    #
    def _reduction_layer(self,ch,maxpool,dilation):
        if maxpool:
            kernel_size=2
            pool_padding=same_padding(kernel_size,dilation)
            return nn.MaxPool2d(
                kernel_size=kernel_size, 
                stride=self.output_stride,
                dilation=dilation,
                padding=pool_padding)
        else:
            return SeparableConv2d(
                in_ch=ch,
                out_ch=ch,
                stride=self.output_stride,
                dilation=dilation)



