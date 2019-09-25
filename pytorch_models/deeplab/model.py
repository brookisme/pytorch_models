import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_models.helpers import output_activation
from pytorch_models.blocks import Conv
import pytorch_models.deeplab.blocks as blocks
from pytorch_models.xception.model import Xception
from pytorch_models.resnet.model import Resnet



class DeeplabV3plus(nn.Module):
    r""" (*) DeeplabV3+

    * Defaults reproduced DeeplabV3+
    - can add more upsampling steps
    - can add a final "refinement_conv" after last upsampling

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
        low_level_out_chs<int|None>:
            - out_chs for low_level_features in backbone
            - if None attempts to get value from backbone-instance
        backbone_out_ch<int>:
            - out_ch from backbone
            - if None attempts to get value from backbone-instance        
        reducer_out_chs<int|float|list|None>
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
        classifier_position<str>: 
            - determines is the classifer comes before or after the final upsample
            - values are 'before' or 'after'
        classifier_depth<int>:
            - depth for classifier block (can be overwritten by classifier config)
        classifier_config<dict>:
            - kwarg dict for Conv Block
        out_activation<str|func|False|None>: 
            - activation method or method name
            - False: No output activation
            - None: Sigmoid if out_ch=1 otherwise Softmax
        out_activation_config<dict>: kwarg-dict for output_activation
    """
    XCEPTION='xception'
    RESNET='resnet'
    UPMODE='bilinear'
    NOT_IMPLEMENTED="Currently only supports 'xception' and 'resnet' backbone"
    BEFORE_UP='before'
    AFTER_UP='after'


    """TODO
    - StrideManger LLO now has steps. need to fix dlv3+ res/xcept to use
    - fix out_chs 
    """
    def __init__(self,
            in_ch,
            out_ch,
            up_chs=256,
            backbone=XCEPTION,
            backbone_config={
                'output_stride': 16,
                'stride_states': [4]
            },
            backbone_low_level_out_chs=None,
            backbone_scale_factors=None,
            backbone_out_ch=None,
            reducer_out_chs=48,
            aspp=True,
            aspp_out_ch=256,
            aspp_config={},
            dropout=None,
            upsample_mode=UPMODE,
            refinement_conv_depth=2,
            refinement_conv_config={},
            classifier_position=BEFORE_UP,
            classifier_depth=1,
            classifier_config={},            
            out_activation=False,
            out_activation_config={}):
        super(DeeplabV3plus,self).__init__()
        self.upsample_mode=upsample_mode
        if dropout is not None:
            backbone_config['dropout']=dropout
            aspp_config['dropout']=dropout
        self._setup_backbone(
            backbone,
            in_ch,
            backbone_config,
            backbone_out_ch,
            backbone_low_level_out_chs,
            backbone_scale_factors)
        if aspp:
            self.aspp=blocks.ASPP(in_ch=self.bb_out_ch,out_ch=aspp_out_ch,**aspp_config)
            refinement_in_ch=aspp_out_ch
        else:
            self.aspp=False
            refinement_in_ch=self.bb_out_ch
        reducer_out_chs=self._reducer_out_chs(reducer_out_chs)
        self.reducers=self._reducers(reducer_out_chs)
        up_chs=self._up_chs(up_chs)
        self.refinement_convs=self._refinement_convs(
            refinement_in_ch,
            reducer_out_chs,
            up_chs,
            dropout,
            refinement_conv_depth,
            refinement_conv_config)
        self.classifier_position=classifier_position
        self.classifier=self._classifier(up_chs[-1],out_ch,classifier_depth,classifier_config)
        self.act=output_activation(out_activation,out_activation_config)

            
    def forward(self,x):
        x,lowxs=self.backbone(x)
        if self.aspp:
            x=self.aspp(x)
        for lx,ch,red,sf,ref in zip(
                lowxs,
                self.low_level_out_chs,
                self.reducers,
                self.scale_factors[:-1],
                self.refinement_convs):
            x=self._up(x,scale_factor=sf)
            lx=red(lx)
            x=torch.cat([x,lx],dim=1)
            if ref:
                x=ref(x)
        if self.classifier_position==DeeplabV3plus.BEFORE_UP:
            x=self.classifier(x)
        if self.scale_factors[-1]>1:
            x=self._up(x,scale_factor=self.scale_factors[-1])
        if self.classifier_position==DeeplabV3plus.AFTER_UP:
            x=self.classifier(x)
        if self.act:
            x=self.act(x)
        return x



    #
    # INTERNAL
    #
    def _setup_backbone(self,
            backbone,
            in_ch,
            backbone_config,
            out_ch,
            low_level_out_chs,
            scale_factors):
        if backbone==DeeplabV3plus.XCEPTION:
            self.backbone=Xception(in_ch=in_ch,**backbone_config)
        elif backbone==DeeplabV3plus.RESNET:
            self.backbone=Resnet(in_ch=in_ch,**backbone_config)
        elif not isinstance(backbone,nn.Module):
            self.backbone=None
        if self.backbone:
            self.bb_out_ch=out_ch or self.backbone.out_ch
            self.low_level_out_chs=low_level_out_chs or self.backbone.low_level_channels
            self.scale_factors=scale_factors or self.backbone.scale_factors
        else:
            raise NotImplementedError(NOT_IMPLEMENTED)


    def _reducer_out_chs(self,reducer_out_chs):
        if isinstance(reducer_out_chs,int):
            reducer_out_chs=[reducer_out_chs]*len(self.low_level_out_chs)
        elif isinstance(reducer_out_chs,float):
            reducer_out_chs=[int(reducer_out_chs*ch) for ch in self.low_level_out_chs]
        elif reducer_out_chs:
            reducer_out_chs=reducer_out_chs[::-1]
        else:
            reducer_out_chs=[False]*len(self.low_level_out_chs)
        return reducer_out_chs


    def _reducers(self,reducer_out_chs):
        reducers=[]
        for in_ch,out_ch in zip(self.low_level_out_chs,reducer_out_chs):
            if (in_ch==out_ch) or (not out_ch):
                reducers.append(nn.Identity())
            else:
                reducers.append(nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=1))
        return nn.ModuleList(reducers)


    def _up_chs(self,up_chs):
        if isinstance(up_chs,int):
            up_chs=[up_chs]*len(self.low_level_out_chs)
        elif not up_chs:
            up_chs=self.low_level_out_chs
        return up_chs


    def _refinement_convs(self,
            encoder_out_ch,
            reducer_out_chs,
            up_chs,
            dropout,
            depth,
            config):
        refiners=[]
        if config.get('dropout') is None:
            config['dropout']=dropout
        if config.get('depth') is None:
            config['depth']=depth
        feat_ins=[encoder_out_ch]+up_chs
        rl_outs=[ r or l for r,l in zip(reducer_out_chs,self.low_level_out_chs)]
        in_chs=[ f+rl for f,rl in zip(feat_ins,rl_outs)]
        for in_ch,out_ch in zip(in_chs,up_chs):
            if out_ch:
                refiners.append(Conv(in_ch=in_ch,out_ch=out_ch,**config))
            else:
                refiners.append(False)
        return nn.ModuleList(refiners)


    def _classifier(self,in_ch,out_ch,depth,config):
        config['depth']=config.get('depth',depth)
        return Conv(in_ch=in_ch,out_ch=out_ch,**config)


    def _up(self,x,scale_factor=4):
        return F.interpolate(
                x, 
                scale_factor=scale_factor, 
                mode=self.upsample_mode, 
                align_corners=True)




