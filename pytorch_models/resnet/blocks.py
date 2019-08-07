import torch
import torch.nn as nn
import pytorch_models.blocks as blocks


class ResBlock(nn.Module):
    """ ResBlock

    """
    def __init__(self,
            in_ch,
            nb_blocks,
            init_stride=None,
            shortcut_method=blocks.Residual.CONV_SHORTCUT,
            **conv_config):
        super(ResBlock, self).__init__()
        self._init_properties(in_ch,shortcut_method)
        self.blocks=self._blocks(in_ch,nb_blocks,init_stride,conv_config)
        self.out_ch=self.blocks[-1].out_ch


    def forward(self,x):
        return self.blocks(x)


    #
    # INTERNAL
    #
    def _init_properties(self,in_ch,shortcut_method):
        self.in_ch=in_ch
        self.shortcut_method=shortcut_method
        self.output_stride=1


    def _blocks(self,in_ch,nb_blocks,init_stride,conv_config):
        layers=[]
        for i in range(nb_blocks):
            if i==0 and (init_stride!=1):
                strides=[init_stride]+[1]*(self._block_depth(conv_config)-1)
                self.output_stride=init_stride
            else:
                strides=None
            rblock=blocks.Residual(
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
            return self.shortcut_method
        else:
            out_chs=config.get('out_chs')
            if out_chs:
                out_ch=out_chs[-1]
            else:
                out_ch=config.get('out_ch',in_ch)
            delta_ch=out_ch-in_ch
            if delta_ch:
                return self.shortcut_method
            else:
                return blocks.Residual.IDENTITY_SHORTCUT



