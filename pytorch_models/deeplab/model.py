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
        backbone<str>: 
            - name of backbone
            - one of ['xception']
        backbone_config<dict>:
            - kwarg-dict for backbone
            - in_ch set explicitly above
        self.low_level_out_chs<int|None>:
            - out_chs for low_level_features in backbone
            - if None attempts to get value from backbone-instance
        backbone_out_ch<int>:
            - out_ch from backbone
            - if None attempts to get value from backbone-instance        
        reduced_low_level_out_ch: 
            - the number of channels to reduce low-level-features to
            - before concatenation
        aspp_out_ch=256,
        aspp_out_ch<int>: number of out_channels from ASPP block
        aspp_config<dict>:
            - kwarg-dict from aspp block
            - aspp_out_ch set explicitly above
        dropout<float|bool|None>:
            - global dropout
            - if not None overrides dropout in backbone/aspp_config
            - if is True dropout=0.5
        upsample_mode<str>: upsampling mode
        refinement_conv_depth<int>:
            - depth for refinement_conv block
        refinement_conv_config<dict>:
            - kwarg-dict for Conv block
        out_activation<str|func|False|None>: 
            - activation method or method name
            - False: No output activation
            - None: Sigmoid if out_ch=1 otherwise Softmax
        out_activation_config<dict>: kwarg-dict for output_activation
    """
    XCEPTION='xception'
    RESNET='resnet'
    UPMODE='bilinear'
    LOW_LEVEL_OUTPUT='half'
    NOT_IMPLEMENTED="Currently only supports 'xception' and 'resnet' backbone"


    def __init__(self,
            in_ch,
            out_ch,
            backbone=XCEPTION,
            backbone_config={
                'output_stride': 16,
                'low_level_output': LOW_LEVEL_OUTPUT
            },
            backbone_low_level_out_chs=None,
            backbone_scale_factors=None,
            backbone_out_ch=None,
            reduced_low_level_out_ch=128,
            aspp_out_ch=256,
            aspp_config={},
            dropout=None,
            upsample_mode=UPMODE,
            refinement_conv_depth=2,
            refinement_conv_config={},
            out_activation=False,
            out_activation_config={}):
        super(DeeplabV3plus,self).__init__()
        self.upsample_mode=upsample_mode
        if dropout is not None:
            backbone_config['dropout']=dropout
            aspp_config['dropout']=dropout
        self.backbone,bb_out_ch,self.low_level_out_chs,self.scale_factors=self._backbone(
            backbone,
            in_ch,
            backbone_config,
            backbone_out_ch,
            backbone_low_level_out_chs,
            backbone_scale_factors)
        self.aspp=blocks.ASPP(in_ch=bb_out_ch,out_ch=aspp_out_ch,**aspp_config)
        self.reducers=self._reducers(reduced_low_level_out_ch)
        self.refinement_conv=self._refinement_conv(
            aspp_out_ch+reduced_low_level_out_ch,
            out_ch,
            dropout,
            refinement_conv_depth,
            refinement_conv_config)
        self.act=output_activation(out_activation,out_activation_config)


    def _reducers(self,reduced_low_level_out_ch):
        reducers=[]
        for out_ch in self.low_level_out_chs:
            reducers.append(nn.Conv2d(
                in_channels=out_ch,
                out_channels=reduced_low_level_out_ch,
                kernel_size=1))
        return nn.ModuleList(reducers)

            
    def forward(self,x):
        x,lowxs=self.backbone(x)
        x=self.aspp(x)
        for lx,ch,r,sf in zip(
                lowxs,
                self.low_level_out_chs,
                self.reducers,
                self.scale_factors[:-1]):
            x=self._up(x,scale_factor=sf)
            lx=r(lx)
            x=torch.cat([x,lx],dim=1)
        if self.refinement_conv:
            x=self.refinement_conv(x)
        x=self._up(x,scale_factor=self.scale_factors[-1])
        if self.act:
            x=self.act(x)
        return x


    def _backbone(self,
            backbone,
            in_ch,
            backbone_config,
            out_ch,
            low_level_out_chs,
            scale_factors):
        if backbone==DeeplabV3plus.XCEPTION:
            backbone=Xception(in_ch=in_ch,**backbone_config)
        elif backbone==DeeplabV3plus.RESNET:
            backbone=Resnet(in_ch=in_ch,**backbone_config)
        elif not isinstance(backbone,nn.Module):
            backbone=None
        if backbone:
            out_ch=out_ch or backbone.out_ch
            low_level_out_chs=low_level_out_chs or backbone.low_level_channels
            scale_factors=scale_factors or backbone.scale_factors
            return backbone, out_ch, low_level_out_chs, scale_factors
        else:
            raise NotImplementedError(NOT_IMPLEMENTED)


    def _refinement_conv(self,in_ch,out_ch,dropout,depth,config):
        if depth:
            if config.get('dropout') is None:
                config['dropout']=dropout
            if config.get('depth') is None:
                config['depth']=depth
            config['in_ch']=in_ch
            config['out_ch']=out_ch
            return Conv(**config)
        else:
            return False


    def _up(self,x,scale_factor=4):
        return F.interpolate(
                x, 
                scale_factor=scale_factor, 
                mode=self.upsample_mode, 
                align_corners=True)




