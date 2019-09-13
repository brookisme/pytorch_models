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
            kernel_sizes=[5,3]
            pooling=False,
            residual=True,
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
                shortcut_method=blocks.Residual.IDENTITY_SHORTCUT)


    def forward(self,x):
        return self.rspp(x)


    def _spp(self,kernel_sizes,pooling,config):
        config['kernel_sizes']=kernel_sizes
        config['pooling']=pooling
        config['dilations']=config.get('dilations',[1]*len(kernel_sizes))
        return ASPP(**config)




