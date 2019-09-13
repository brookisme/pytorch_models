import torch.nn as nn
import pytorch_models.blocks as blocks
from pytorch_models.deeplab.blocks import ASPP



class RSPP(nn.Module):
    """ Residual Spatial Pyramid Pooling
    TODO: docs!
    """

    def __init__(self,
            in_ch,
            out_ch=None,
            kernel_sizes=[5,3],
            pooling=False,
            residual=True,
            shortcut_method=blocks.Residual.AUTO_SHORTCUT,
            spp_config={}):
        super(RSPP, self).__init__()
        if out_ch is None:
            out_ch=in_ch
        self.in_ch=in_ch
        self.out_ch=out_ch
        spp=self._spp(kernel_sizes,pooling,spp_config)
        self.rspp=blocks.Residual(
                in_ch=self.in_ch,
                out_ch=self.out_ch,
                block=spp,
                is_residual_block=residual,
                shortcut_stride=1,
                shortcut_method=shortcut_method)


    def forward(self,x):
        return self.rspp(x)


    def _spp(self,kernel_sizes,pooling,config):
        config['kernel_sizes']=kernel_sizes
        config['pooling']=pooling
        config['dilations']=config.get('dilations',[1]*len(kernel_sizes))
        config['join_method']=ASPP.ADD
        if config.get('out_conv_config') is None:
            config['out_conv_config']=False
        return ASPP(self.in_ch,self.out_ch,**config)




