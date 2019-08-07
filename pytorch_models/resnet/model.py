import torch.nn as nn
import torch.nn.functional as F
import pytorch_models.helpers as h
from pytorch_models.blocks import Residual, Conv
from pytorch_models.resnet.blocks import ResBlock
import pytorch_models.xception.blocks as blocks
import pytorch_models.classifiers as classifiers

#
# CONSTANTS
#
LOW_LEVEL_ERROR='Resnet.low_level_output_strides = {} not implemented'
_preset_models={
    '18': [
        {   
            'init_stride': 1,
            'nb_blocks': 2,
            'conv': { 'out_ch': 64, 'depth': 2 }
        },{
            'nb_blocks': 2,
            'conv': { 'out_ch': 128, 'depth': 2 }
        },{
            'nb_blocks': 2,
            'conv': { 'out_ch': 256, 'depth': 2 }
        },{
            'nb_blocks': 2,
            'conv': { 'out_ch': 512, 'depth': 2 }
        }
    ],
    '34': [
        {   
            'init_stride': 1,
            'nb_blocks': 3,
            'conv': { 'out_ch': 64, 'depth': 3 }
        },{
            'nb_blocks': 4,
            'conv': { 'out_ch': 128, 'depth': 4 }
        },{
            'nb_blocks': 6,
            'conv': { 'out_ch': 256, 'depth': 6 }
        },{
            'nb_blocks': 3,
            'conv': { 'out_ch': 512, 'depth': 3 }
        }
    ],
    '50': [
        {
            'init_stride': 1,
            'nb_blocks': 3,
            'conv': { 'out_chs': [64,64,256], 'kernel_sizes': [1,3,1] }
        },{
            'nb_blocks': 4,
            'conv': { 'out_chs': [128,128,512], 'kernel_sizes': [1,3,1] }
        },{
            'nb_blocks': 6,
            'conv': { 'out_chs': [256,256,1024], 'kernel_sizes': [1,3,1] }
        },{
            'nb_blocks': 3,
            'conv': { 'out_chs': [512,512,2048], 'kernel_sizes': [1,3,1] }
        }
    ],
    '101': [
        {
            'init_stride': 1,
            'nb_blocks': 3,
            'conv': { 'out_chs': [64,64,256], 'kernel_sizes': [1,3,1] }
        },{
            'nb_blocks': 4,
            'conv': { 'out_chs': [128,128,512], 'kernel_sizes': [1,3,1] }
        },{
            'nb_blocks': 23,
            'conv': { 'out_chs': [256,256,1024], 'kernel_sizes': [1,3,1] }
        },{
            'nb_blocks': 3,
            'conv': { 'out_chs': [512,512,2048], 'kernel_sizes': [1,3,1] }
        }
    ],
    '152': [
        {
            'init_stride': 1,
            'nb_blocks': 3,
            'conv': { 'out_chs': [64,64,256], 'kernel_sizes': [1,3,1] }
        },{
            'nb_blocks': 8,
            'conv': { 'out_chs': [128,128,512], 'kernel_sizes': [1,3,1] }
        },{
            'nb_blocks': 36,
            'conv': { 'out_chs': [256,256,1024], 'kernel_sizes': [1,3,1] }
        },{
            'nb_blocks': 3,
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
        low_level_output_strides<int|list|bool>:
            *** 
        dropout<float|bool|None>: 
            * TODO: NOT IMPLEMENTEDED
            - global dropout
            - if is True dropout=0.5
        nb_classes<int|None>:
            - None for backbone use (no classifier added)
            - or list of residual block configurations
        classifier<str|nn.Module>: alias, module-name or module for classifier
        classifier_config<dict>: 
            - kwarg-dict for classifier
            - nb_classes/in_ch will be derived from xception architecture
    """
    MODELS=_preset_models
    DEFAULT_INPUT_CONV={ 'out_ch': 64, 'kernel_size': 7, 'stride': 2 }
    DEFUALT_INPUT_POOL={ 'kernel_size': 3, 'stride': 2 }
    LOW_LEVEL_ALL='all'
    LOW_LEVEL_RES='resblock'
    LOW_LEVEL_INPUT_CONV='input_conv'
    LOW_LEVEL_POOL='pool'
    #
    # INSTANCE METHODS
    #
    def __init__(self,
            in_ch,
            input_conv=DEFAULT_INPUT_CONV,
            input_pool=DEFUALT_INPUT_POOL,
            blocks=18,
            shortcut_method=Residual.CONV_SHORTCUT,
            low_level_output_strides=False,
            dropout=False,
            nb_classes=None,
            classifier='gap',
            classifier_config={}):
        super(Resnet,self).__init__()
        self._init_properties(in_ch,shortcut_method,low_level_output_strides)
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
        self._init_output_stride_state()
        low_level_features=[]
        low_level_channels=[]
        if self.input_conv:
            x=self.input_conv(x)
            if self._is_low_level_feature(Resnet.LOW_LEVEL_INPUT_CONV):
                low_level_features.append(x)
                low_level_channels.append(self.input_conv.out_ch)
        if self.input_pool:
            x=self.input_pool(x)
            if self._is_low_level_feature(Resnet.LOW_LEVEL_POOL):
                low_level_features.append(x)
                low_level_channels.append(self.input_conv.out_ch)    
        for block in self.blocks:
            x=block(x)
            if block.output_stride and self._is_low_level_feature(Resnet.LOW_LEVEL_RES):
                low_level_features.append(x)
                low_level_channels.append(block.out_ch)
        if self.classifier_block:
            return self.classifier_block(x)
        elif low_level_channels:
            return x, low_level_features, low_level_channels
        else:
            return x


    #
    # INTERNAL
    #
    def _init_properties(self,in_ch,shortcut_method,low_level_output_strides):
        self.in_ch=in_ch
        self.default_shortcut_method=shortcut_method
        self.low_level_output_strides=low_level_output_strides


    def _blocks(self,in_ch,blocks):
        layers=[]
        for block in self._blocks_list(blocks):
            block_config, conv_config=self._parse_block(block)
            rblock=ResBlock(in_ch,**block_config,**conv_config)
            layers.append(rblock)
            in_ch=rblock.out_ch
        return nn.ModuleList(layers)


    def _parse_block(self,block):
        block_config=block.copy()        
        block_config['shortcut_method']=block.get(
            'shortcut_method',
            self.default_shortcut_method)
        block_config['init_stride']=block.get('init_stride',2)
        conv_config=block_config.pop('conv')
        return block_config, conv_config


    def _blocks_list(self,blocks):
        if isinstance(blocks,int):
            blocks=str(blocks)
        if isinstance(blocks,str):
            blocks=Resnet.MODELS[blocks]
        return blocks


    def _init_output_stride_state(self):
        self.output_stride_index=0
        self.output_stride_state=None


    def _increment_output_stride_state(self):
        self.output_stride_index+=1
        self.output_stride_state=(2**self.output_stride_index)


    def _is_low_level_feature(self,tag=None):
        self._increment_output_stride_state()
        if self.low_level_output_strides:
            if isinstance(self.low_level_output_strides,int):
                return self.low_level_output_strides==self.output_stride_state
            elif isinstance(self.low_level_output_strides,str):
                return self.low_level_output_strides==tag
            else:
                state_is_in=self.output_stride_state in self.low_level_output_strides
                if tag:
                    tag_is_in=tag in self.low_level_output_strides
                    return state_is_in or tag_is_in
                else:
                    return state_is_in
        else:
            return False






