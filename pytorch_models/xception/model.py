import torch.nn as nn
import torch.nn.functional as F
from pytorch_models.helpers import LowLevelFeatures
import pytorch_models.xception.blocks as blocks
import pytorch_models.classifiers as classifiers



class Xception(nn.Module):
    r""" (Modified Aligned) Xception

    Xception Network or Backbone from DeeplabV3+
    
    Args:
        in_ch<int>: number of input channels
        output_stride<int>: output_stride
        low_level_output<int|list|str|False>:
            - if truthy return low_level_features and low_level_channels
            - preset options exclude the last block
            - options:
                - Xception.LOW_LEVEL_ALL: after all blocks
                - Xception.LOW_LEVEL_XBLOCK: after all (downsampling) xception-blocks
                - Xception.LOW_LEVEL_UNET: use for unet ( after entry_conv and xception-blocks )
                - Xception.LOW_LEVEL_ENTRY: after entry_conv
                - A list containing output strides of interest and/or 
                  any of the above strings
        low_level_drop_array<bool>: 
            - if True and len(low-level-outputs)==1 return value instead of array
        entry_ch: 32
        entry_out_ch: 64
        xblock_chs: [128,256,728]
        bottleneck_depth: 16
        exit_xblock_ch: 1024
        exit_stack_chs: [1536,1536,2048]
        xblock_depth: 3
        dropout<float|bool|None>:
            - global dropout
            - if is True dropout=0.5
        nb_classes<int|None>:
            - None for backbone use (no classifier added)
        classifier<str|nn.Module>: alias, module-name or module for classifier
        classifier_config<dict>: 
            - kwarg-dict for classifier
            - nb_classes/in_ch will be derived from xception architecture
    """
    LOW_LEVEL_XBLOCK='xblock'
    LOW_LEVEL_UNET='unet'
    LOW_LEVEL_ENTRY='entry'
    #
    # INSTANCE METHODS
    #
    def __init__(self,
            in_ch,
            output_stride=None,
            low_level_output=False,
            low_level_drop_array=True,
            entry_ch=32,
            entry_out_ch=64,
            xblock_chs=[128,256,728],
            bottleneck_depth=16,
            exit_xblock_ch=1024,
            exit_stack_chs=[1536,1536,2048],
            xblock_depth=3,
            dropout=False,
            nb_classes=None,
            classifier='gap',
            classifier_config={}):
        super(Xception,self).__init__()
        self.output_stride=output_stride
        self.low_level_output=low_level_output
        self.low_level_drop_array=low_level_drop_array
        self.dropout=dropout
        llf=LowLevelFeatures(
            self.output_stride,
            self.low_level_output,
            drop_array=self.low_level_drop_array)
        self.entry_block=blocks.EntryBlock(in_ch,entry_ch,entry_out_ch)
        llf.increment(self.entry_block.output_stride)
        self.xblocks=self._xblocks(
            entry_out_ch,
            xblock_chs,
            xblock_depth,llf)
        self.bottleneck=blocks.SeparableStack(
            in_ch=xblock_chs[-1],
            depth=bottleneck_depth,
            res=True,
            dilation=llf.dilation,
            dropout=self.dropout)
        self.exit_xblock=blocks.XBlock(
                in_ch=xblock_chs[-1],
                out_ch=exit_xblock_ch,
                depth=xblock_depth,
                dilation=llf.dilation,
                dropout=self.dropout)
        llf.increment(self.exit_xblock.output_stride)
        self.exit_stack=blocks.SeparableStack(
            in_ch=exit_xblock_ch,
            out_chs=exit_stack_chs,
            dilation=llf.dilation,
            dropout=self.dropout)
        if nb_classes:
            classifier_config['nb_classes']=nb_classes
            classifier_config['in_ch']=exit_stack_chs[-1]
            classifier=classifiers.get(classifier)
            self.classifier_block=classifier(**classifier_config)
        else:
            self.classifier_block=False
            self.out_ch=exit_stack_chs[-1]


    def forward(self,x):
        llf=LowLevelFeatures(
            self.output_stride,
            self.low_level_output,
            drop_array=self.low_level_drop_array)
        x=self.entry_block(x)
        llf.increment(self.entry_block.output_stride)
        llf.update_low_level_features(
                x,
                self.entry_block.out_ch,
                Xception.LOW_LEVEL_ENTRY)
        for xblock in self.xblocks:
            x=xblock(x)
            llf.increment(xblock.output_stride)
            llf.update_low_level_features(
                    x,
                    xblock.out_ch,
                    Xception.LOW_LEVEL_XBLOCK)
        x=self.bottleneck(x)
        x=self.exit_xblock(x)
        x=self.exit_stack(x)
        if self.classifier_block:
            return self.classifier_block(x)
        elif llf.low_level_output:
            return (x, *llf.out())
        else:
            return x


    #
    # INTERNAL
    #
    def _xblocks(self,in_ch,out_ch_list,depth,llf):
        layers=[]
        for ch in out_ch_list:
            block=blocks.XBlock(
                in_ch,
                out_ch=ch,
                depth=depth,
                dilation=llf.dilation,
                dropout=self.dropout
            )
            layers.append(block)
            llf.increment(block.output_stride)
            in_ch=ch
        return nn.ModuleList(layers)


    def _init_stride_state(self):
        llf.dilation=1
        self.stride_index=0
        self.stride_state=None
        self.at_low_level_stride=False


    def _increment_stride_state(self):
        self.stride_index+=1
        self.stride_state=(2**self.stride_index)
        self.at_low_level_stride=(self.stride_state==self.low_level_stride)
        if self.stride_state>=self.output_stride:
            llf.dilation*=2




