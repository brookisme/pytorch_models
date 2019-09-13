import torch.nn as nn
from pytorch_models.blocks import Residual
from pytorch_models.unet.blocks import RSPP
from pytorch_models.helpers import same_padding



class UNet(nn.Module):
    """ (Modified) UNet

    A UNet that allows you to include ...

    Args:



    """
    STRIDE='stride'
    MAX_POOL='max_pool'
    def __init__(self,
            in_ch,
            out_ch,
            depth=5,
            down_ch=64,
            down_chs=None,
            down_method=STRIDE,
            skip_indices=None,
            up_chs=None,
            refinement_reducer=False,
            refinement_chs=False,
            max_pool_kernel_size=3,
            down={
                'rspp': False,
                'squeeze_excitation': False,
                'depth': 2,
                'is_residual_block': True,
            },
            bottleneck={
                'depth': 5,
                'is_residual_block': True
            },
            up={
                'rspp': False,
                'squeeze_excitation': False,
                'depth': 2,
                'is_residual_block': True,
            }
        ):
        super(UNet,self).__init__()
        self.in_ch=in_ch
        self.out_ch=out_ch
        self.max_pool_kernel_size=max_pool_kernel_size
        if not down_chs:
            self.down_chs=[down_ch*(2**n) for n in range(depth)]
        self.down_blocks=self._down_blocks(in_ch,down,down_method)
        if bottleneck:
            self.bottleneck=self._building_block(self.down_chs[-1],**bottleneck)
        else:
            self.bottleneck=False
        if skip_indices is None:
            skip_indices=list(range(self.down_chs))
        if skip_indices: 
            skip_indices.sort()
        self.skip_indices=skip_indices
        if not up_chs:
            up_chs=[self.down_chs[i] for i in skip_indices[::-1]]
        self.up_chs=up_chs
        self.up_blocks=self._up_blocks(in_ch,up)
        self.refinements=self._refinements(refinement_reducer,refinement_chs)


    def _refinements(self,reducer,out_chs):
        if (not out_chs) and isinstance(reducer,float):
            out_chs=[reducer*self.down_chs[i] for i in skip_indices[::-1]]
        if out_chs:
            refinements=[]
            in_ch=self.down_chs[-1]
            for out_ch in out_chs:
                refinements.append(nn.Conv2d(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=1))
                in_ch=out_ch
        else:
            refinements=[False]*len(self.skip_indices)
        return refinements


    def _down_blocks(self,in_ch,config,down_method):
        blocks=[]
        blocks.append(self._building_block(in_ch,self.down_chs[0],config))
        in_ch=self.down_chs[0]          
        for out_ch in self.down_chs[1:]:
            blocks.append(self._building_block(in_ch,out_ch,config,down_method))
            in_ch=out_ch
        return nn.ModuleList(blocks)


    def _up_blocks(self,in_ch,config):
        blocks=[]    
        for out_ch in self.up_chs:
            blocks.append(self._building_block(in_ch,out_ch,config))
            in_ch=out_ch
        return nn.ModuleList(blocks)


    def _building_block(self,in_ch,out_ch=None,config={},down_method=False):
        blocks=[]
        config=config.copy()
        rspp=config.pop('rspp',False)
        squeeze_excitation=config.pop('squeeze_excitation',False)
        if down_method==UNet.STRIDE:
            config['stride']=2
        elif down_method==MAX_POOL:
            blocks.append(n.MaxPool2d(
                kernel_size=self.max_pool_kernel_size, 
                stride=2,
                padding=same_padding(self.max_pool_kernel_size)))
        elif down_method:
            raise ValueError('down_method values: stride, max_pool, False|None')
        if rspp:
            blocks.append(RSPP(in_ch,out_ch,**config))
        else:
            blocks.append(Residual(in_ch,out_ch,**config))            
        if squeeze_excitation:
            se="TODO"
            blocks.append(se)
        return nn.Sequential(*blocks)


    def forward(self,x):
        skips=[]
        stride_states=[]
        current_stride_state=1
        
        for i,dblock in enumerate(self.down_blocks):
            x=xblock(x)
            if i: current_stride_state*=2
            if i in self.skip_indices:
                skips.append(x)
                stride_states.append(current_stride_state)

        if self.bottleneck:
            x=self.bottleneck(x)
        
        for skip,stride_state,rfine,ublock in zip(
                skips,
                stride_states,
                self.refinements,
                self.up_blocks):
            up_factor=current_stride_state/stride_state
            x=self._up(x,up_factor)
            if rfine:
                skip=rfine(skip)
            x=torch.cat([x,skip])
            x=ublock(x)
            current_stride_state=stride_state

        if  current_stride_state!=1:           
            up_factor=current_stride_state/stride_state
            x=self._up(x,up_factor)

        x=self.out_block(x)

        if self.act:
            x=self.act(x)

        return x

