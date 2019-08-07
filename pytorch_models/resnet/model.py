import torch.nn as nn
import torch.nn.functional as F
import pytorch_models.helpers as h
from pytorch_models.blocks import Residual, Conv
import pytorch_models.xception.blocks as blocks
import pytorch_models.classifiers as classifiers


_preset_models={
    '18': [
        {   
            'init_stride': 1,
            'depth': 2,
            'conv': { 'out_ch': 64, 'depth': 2 }
        },            
        {   
            'depth': 2,
            'conv': { 'out_ch': 128, 'depth': 2 }
        },            
        {   
            'depth': 2,
            'conv': { 'out_ch': 256, 'depth': 2 }
        },            
        {   
            'depth': 2,
            'conv': { 'out_ch': 512, 'depth': 2 }
        }
    ],
    '34': [
        {   
            'init_stride': 1,
            'depth': 2,
            'conv': { 'out_ch': 64, 'depth': 3 }
        },            
        {   
            'depth': 2,
            'conv': { 'out_ch': 128, 'depth': 4 }
        },            
        {   
            'depth': 2,
            'conv': { 'out_ch': 256, 'depth': 6 }
        },            
        {   
            'depth': 2,
            'conv': { 'out_ch': 512, 'depth': 3 }
        }
    ],
    '50': [
        {
            'init_stride': 1,
            'depth': 3,
            'conv': { 'out_chs': [64,64,256], 'kernel_sizes': [1,3,1] }
        },{
            'depth': 4,
            'conv': { 'out_chs': [128,128,512], 'kernel_sizes': [1,3,1] }
        },{
            'depth': 6,
            'conv': { 'out_chs': [256,256,1024], 'kernel_sizes': [1,3,1] }
        },{
            'depth': 3,
            'conv': { 'out_chs': [512,512,2048], 'kernel_sizes': [1,3,1] }
        }
    ],
    '101': [
        {
            'init_stride': 1,
            'depth': 3,
            'conv': { 'out_chs': [64,64,256], 'kernel_sizes': [1,3,1] }
        },{
            'depth': 4,
            'conv': { 'out_chs': [128,128,512], 'kernel_sizes': [1,3,1] }
        },{
            'depth': 23,
            'conv': { 'out_chs': [256,256,1024], 'kernel_sizes': [1,3,1] }
        },{
            'depth': 3,
            'conv': { 'out_chs': [512,512,2048], 'kernel_sizes': [1,3,1] }
        }
    ],
    '152': [
        {
            'init_stride': 1,
            'depth': 3,
            'conv': { 'out_chs': [64,64,256], 'kernel_sizes': [1,3,1] }
        },{
            'depth': 8,
            'conv': { 'out_chs': [128,128,512], 'kernel_sizes': [1,3,1] }
        },{
            'depth': 36,
            'conv': { 'out_chs': [256,256,1024], 'kernel_sizes': [1,3,1] }
        },{
            'depth': 3,
            'conv': { 'out_chs': [512,512,2048], 'kernel_sizes': [1,3,1] }
        }
    ]

}

class Resnet(nn.Module):
    r""" Resnet https://arxiv.org/pdf/1512.03385.pdf

    Resnet Network or Backbone from DeeplabV3+
    
    Args:
        in_ch<int>: number of input channels
        blocks<str|int|<list<dict>>: 
            - model configuration key (if int convert to str)
            - resnet-blocks configuration list
        shortcut_method<None|str>
            - method for managing shortcuts with increasing dimensions. options:
            - Residual.IDENTITY_SHORTCUT | None: use identity
            - Residual.ZERO_PADDING_SHORTCUT: add zero padding
            - Residual.CONV_SHORTCUT: use a 1x1 conv to increase the padding
        nb_classes<int|None>:
            - None for backbone use (no classifier added)
            - or list of residual block configurations
        dropout<float|bool|None>:
            - global dropout
            - if is True dropout=0.5
        classifier<str|nn.Module>: alias, module-name or module for classifier
        classifier_config<dict>: 
            - kwarg-dict for classifier
            - nb_classes/in_ch will be derived from xception architecture
    """
    MODELS=_preset_models
    DEFAULT_INPUT_CONV={ 'out_ch': 64, 'kernel_size': 7, 'stride': 2 }
    DEFUALT_INPUT_POOL={ 'kernel_size': 2 }
    #
    # INSTANCE METHODS
    #
    def __init__(self,
            in_ch,
            input_conv=DEFAULT_INPUT_CONV,
            input_pool=DEFUALT_INPUT_POOL,
            blocks=18,
            shortcut_method=Residual.CONV_SHORTCUT,
            dropout=False,
            nb_classes=None,
            classifier='gap',
            classifier_config={}):
        super(Resnet,self).__init__()
        self.default_shortcut_method=shortcut_method
        if input_conv:
            self.input_conv=Conv(in_ch,**input_conv)
            in_ch=self.input_conv.out_ch
        else:
            self.input_conv=False
        if input_pool:
            self.input_pool=nn.MaxPool2d(**input_pool)
        else:
            self.input_pool=False
        self.blocks=self._blocks(in_ch,blocks)
        blocks_out_ch=self.blocks[-1].out_ch
        if nb_classes:
            classifier_config['nb_classes']=nb_classes
            classifier_config['in_ch']=blocks_out_ch
            classifier=classifiers.get(classifier)
            self.classifier_block=classifier(**classifier_config)
        else:
            self.classifier_block=False
            self.out_ch=blocks_out_ch


    def forward(self,x):
        if self.input_conv:
            x=self.input_conv(x)
        if self.input_pool:
            x=self.input_pool(x)
        x=self.blocks(x)
        if self.classifier_block:
            return self.classifier_block(x)
        else:
            return x


    #
    # INTERNAL
    #
    def _blocks(self,in_ch,blocks):
        layers=[]
        for block in self._blocks_list(blocks):
            conv_config=block['conv']
            depth=block['depth']
            init_stride=block.get('init_stride',2)
            for i in range(depth):
                if i==0 and (init_stride!=1):
                    strides=[init_stride]+[1]*(self._block_depth(conv_config)-1)
                else:
                    strides=None
                rblock=Residual(
                    in_ch=in_ch,
                    strides=strides,
                    shortcut_method=self._shortcut_method(
                        in_ch,
                        conv_config,
                        strides is not None),
                    shortcut_stride=init_stride,
                    **conv_config
                )
                in_ch=rblock.out_ch
                layers.append(rblock)
        return nn.Sequential(*layers)


    def _blocks_list(self,blocks):
        if isinstance(blocks,int):
            blocks=str(blocks)
        if isinstance(blocks,str):
            blocks=Resnet.MODELS[blocks]
        return blocks


    def _block_depth(self,config):
        depth=config.get('depth',False)
        if not depth:
            out_chs=config.get('out_chs')
            if out_chs:
                depth=len(out_chs)
            else:
                depth=1
        return depth


    def _shortcut_method(self,in_ch,config,strided):
        if strided:
            return self.default_shortcut_method
        else:
            out_chs=config.get('out_chs')
            if out_chs:
                out_ch=out_chs[-1]
            else:
                out_ch=config.get('out_ch',in_ch)
            delta_ch=out_ch-in_ch
            if delta_ch:
                return self.default_shortcut_method
            else:
                return Residual.IDENTITY_SHORTCUT



