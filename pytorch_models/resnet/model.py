import torch.nn as nn
import torch.nn.functional as F
from pytorch_models.tools import StrideManager
from pytorch_models.blocks import Residual, Conv
from pytorch_models.resnet.blocks import ResBlock
import pytorch_models.xception.blocks as blocks
import pytorch_models.classifiers as classifiers
#
# MODEL CONFIG
#
_preset_models={
    '18': [
        {   
            'output_stride': 1,
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
            'output_stride': 1,
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
            'output_stride': 1,
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
            'output_stride': 1,
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
            'output_stride': 1,
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




#
# RESNET MODEL
#
class Resnet(nn.Module):
    r""" Resnet https://arxiv.org/pdf/1512.03385.pdf

    Resnet Network or Backbone from DeeplabV3+
    
    Args:
        in_ch<int>: number of input channels
        blocks<str|int|<list<dict>>: 
            - model configuration key (if int convert to str)
            - resnet-blocks configuration list
        shortcut_method<None|str>
            - method for managing shortcuts with increasing dimensions
            - options:
                - Residual.IDENTITY_SHORTCUT | None: use identity
                - Residual.ZERO_PADDING_SHORTCUT: add zero padding
                - Residual.CONV_SHORTCUT: use a 1x1 conv to increase the padding
        output_stride:
            - output_stride for output of res-blocks
            - if output_stride use dilations after output stride is reached
        low_level_nb_steps:
            - ...
        low_level_output<int|list|str|False>:
            - if truthy return low_level_features and low_level_channels
            - preset options exclude the last block
            - options:
                - Resnet.LOW_LEVEL_ALL: after all blocks
                - Resnet.LOW_LEVEL_RES: after all (downsampling) resnet-blocks
                - Resnet.LOW_LEVEL_UNET: use for unet ( after input_conv and resblocks )
                - Resnet.LOW_LEVEL_INPUT_CONV: after input-conv-block
                - Resnet.LOW_LEVEL_POOL: after input max-pooling
                - A list containing output strides of interest and/or 
                  any of the above strings
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
    DEFUALT_INPUT_POOL={ 'kernel_size': 3, 'stride': 2, 'padding': 1 }
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
            output_stride=None,
            indices=None,
            stride_states=None,
            tags=None,
            dropout=False,
            nb_classes=None,
            classifier='gap',
            classifier_config={}):
        super(Resnet,self).__init__()
        self.in_ch=in_ch
        self.default_shortcut_method=shortcut_method
        self.stride_manager=StrideManager(output_stride,indices,tags,stride_states)
        if input_conv:
            self.input_conv=Conv(in_ch,**input_conv)
            in_ch=self.input_conv.out_ch
            self.input_conv_stride=input_conv.get('stride',1)
            self.stride_manager.update(
                channels=self.input_conv.out_ch,
                stride=self.input_conv_stride,
                tag=Resnet.LOW_LEVEL_INPUT_CONV)
        else:
            self.input_conv=False
        if input_pool:
            self.input_pool=nn.MaxPool2d(**input_pool)
            self.input_pool_stride=input_pool.get('stride',1)
            self.stride_manager.update(
                channels=self.input_conv.out_ch,
                stride=self.input_pool_stride,
                tag=Resnet.LOW_LEVEL_POOL)
        else:
            self.input_pool=False
        self.blocks=self._blocks(in_ch,blocks)
        self.nb_resnet_blocks=len(self.blocks)
        blocks_out_ch=self.blocks[-1].out_ch
        self.low_level_channels=self.stride_manager.channels()
        self.scale_factors=self.stride_manager.scale_factors()
        if nb_classes:
            classifier_config['nb_classes']=nb_classes
            classifier_config['in_ch']=blocks_out_ch
            classifier=classifiers.get(classifier)
            self.classifier_block=classifier(**classifier_config)
        else:
            self.classifier_block=False
            self.out_ch=blocks_out_ch


    def forward(self,x):
        features=[]
        self.stride_manager.start()
        if self.input_conv:
            x=self.input_conv(x)
            if self.stride_manager.step():
                features.append(x)
        if self.input_pool:
            x=self.input_pool(x)
            if self.stride_manager.step():
                features.append(x)
        for i,block in enumerate(self.blocks,start=1):
            x=block(x)
            if (i!=self.nb_resnet_blocks):
                if self.stride_manager.step():
                    features.append(x)
        if self.classifier_block:
            return self.classifier_block(x)
        elif features:
            return x, features[::-1]
        else:
            return x


    #
    # INTERNAL
    #
    def _blocks(self,in_ch,blocks):
        layers=[]
        for i,block in enumerate(self._blocks_list(blocks)):
            block_config, conv_config, output_stride=self._parse_block(block)
            output_stride=self.stride_manager.stride(output_stride)
            rblock=ResBlock(
                in_ch,
                output_stride=output_stride,
                dilation=self.stride_manager.dilation,
                **block_config,
                **conv_config)
            layers.append(rblock)
            tag="{}_{}".format(Resnet.LOW_LEVEL_RES,i)
            self.stride_manager.update(
                channels=rblock.out_ch,
                stride=rblock.output_stride,
                tag=tag)
            in_ch=rblock.out_ch
        return nn.ModuleList(layers)


    def _parse_block(self,block):
        block_config=block.copy()        
        block_config['shortcut_method']=block.get(
            'shortcut_method',
            self.default_shortcut_method)
        conv_config=block_config.pop('conv')
        output_stride=block_config.pop('output_stride',2)
        return block_config, conv_config, output_stride


    def _blocks_list(self,blocks):
        if isinstance(blocks,int):
            blocks=str(blocks)
        if isinstance(blocks,str):
            blocks=Resnet.MODELS[blocks]
        return blocks






