import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_models.helpers import output_activation
from pytorch_models.blocks import Conv
import pytorch_models.deeplab.blocks as blocks
from pytorch_models.xception.model import Xception
from pytorch_models.resnet.model import Resnet



class DeeplabV3plus(nn.Module):
    r""" DeeplabV3+

    Args:
        in_ch<int>: Number of channels in input
        out_ch<int|None>: 
            - number of channels in output
            - if out_conv_config, specifying out_conv_config['out_ch'] 
              can be used to change final number of output channels
        backbone<str>: 
            - name of backbone
            - one of ['xception']
        backbone_config<dict>:
            - kwarg-dict for backbone
            - in_ch set explicitly above
        backbone_low_level_out_ch<int|None>:
            - out_ch for low_level_features in backbone
            - if None attempts to get value from backbone-instance
        backbone_out_ch<int>:
            - out_ch from backbone
            - if None attempts to get value from backbone-instance        aspp_out_ch=256,
        aspp_out_ch<int>: number of out_channels from ASPP block
        aspp_config<dict>:
            - kwarg-dict from aspp block
            - aspp_out_ch set explicitly above
        dropout<float|bool|None>:
            - global dropout
            - if not None overrides dropout in backbone/aspp_config
            - if is True dropout=0.5
        upsample_mode<str>: upsampling mode
        out_conv_config<dict|False>:
            - kwarg-dict for Conv block
            - False for no out_conv
        out_activation<str|func|False|None>: 
            - activation method or method name
            - False: No output activation
            - None: Sigmoid if out_ch=1 otherwise Softmax
        out_activation_config<dict>: kwarg-dict for output_activation
    """
    XCEPTION='xception'
    RESNET='resnet'
    UPMODE='bilinear'


    def __init__(self,
            in_ch,
            out_ch,
            backbone=XCEPTION,
            backbone_config={},
            backbone_low_level_out_ch=None,
            backbone_out_ch=None,
            aspp_out_ch=256,
            aspp_config={},
            dropout=None,
            upsample_mode=UPMODE,
            out_conv_config=False,
            out_activation=False,
            out_activation_config={}):
        super(DeeplabV3plus,self).__init__()
        self.upsample_mode=upsample_mode
        if dropout is not None:
            backbone_config['dropout']=dropout
            aspp_config['dropout']=dropout
        self.backbone,backbone_out_ch,backbone_low_level_out_ch=self._backbone(
            backbone,
            in_ch,
            backbone_config,
            backbone_out_ch,
            backbone_low_level_out_ch)
        self.aspp=blocks.ASPP(in_ch=backbone_out_ch,out_ch=aspp_out_ch,**aspp_config)
        self.channel_reducer=nn.Conv2d(
            in_channels=aspp_out_ch+backbone_low_level_out_ch,
            out_channels=out_ch,
            kernel_size=1)
        self.out_conv=self._out_conv(out_ch,dropout,out_conv_config)
        self.act=output_activation(out_activation,out_activation_config)


    def forward(self,x):
        x,lowx=self.backbone(x)
        x=self.aspp(x)
        x=self._up(x)
        x=torch.cat([x,lowx],dim=1)
        x=self.channel_reducer(x)
        x=self._up(x)
        if self.out_conv:
            x=self.out_conv(x)
        if self.act:
            x=self.act(x)
        return x


    def _backbone(self,backbone,in_ch,backbone_config,out_ch,low_level_out_ch):
        if backbone==DeeplabV3plus.XCEPTION:
            backbone=Xception(in_ch=in_ch,**backbone_config)
        elif backbone==DeeplabV3plus.RESNET:
            backbone=Resnet(in_ch=in_ch,**backbone_config)
        elif not isinstance(backbone,nn.Module):
            backbone=None
        if backbone:
            out_ch=out_ch or backbone.out_ch
            low_level_out_ch=low_level_out_ch or backbone.low_level_out_ch
            return backbone, out_ch, low_level_out_ch
        else:
            raise NotImplementedError("Currently only supports 'xception' backbone")


    def _out_conv(self,in_ch,dropout,config):
        if config is not False:
            if confg.get('dropout') is None:
                config['dropout']=dropout
            config['in_ch']=in_ch
            return Conv(**config)


    def _up(self,x,scale_factor=4):
        return F.interpolate(
                x, 
                scale_factor=scale_factor, 
                mode=self.upsample_mode, 
                align_corners=True)




