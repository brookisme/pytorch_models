import torch
import torch.nn as nn


class StridedIdentity(nn.Module):
    """ Strided Identity: nn.Conv2d convenience wrapper. 

    Removes pixels between stride, otherwise identity

    Example:
    
    inpt=>
        tensor([[[[  1.,   2.,   3.],
                  [  4.,   5.,   6.],
                  [  7.,   8.,   9.]],

                 [[ 10.,  20.,  30.],
                  [ 40.,  50.,  60.],
                  [ 70.,  80.,  90.]]]])


    StridedIdentity(2)(inpt)=>
        tensor([[[[  1.,   3.],
                  [  7.,   9.]],

                 [[ 10.,  30.],
                  [ 70.,  90.]]]])

    Args:
        ch<int>: number of input/output channels
        stride<int>: stride
    """ 
    #
    # INSTANCE METHODS
    #
    def __init__(self,ch,stride=2):
        super(StridedIdentity, self).__init__()
        self.in_ch=ch
        self.out_ch=ch
        self.strided_eye=self._strided_eye(stride)


    def forward(self, x):
        return self.strided_eye(x)


    #
    # INTERNAL
    #
    def _strided_eye(self,stride):
        ident=torch.nn.Conv2d(
                self.in_ch, 
                self.in_ch, 
                kernel_size=1, 
                stride=stride, 
                groups=self.in_ch, 
                bias=False)
        self._freeze_and_ident(ident)
        return ident


    def _freeze_and_ident(self,ident):
        for p in ident.parameters():
            p.requires_grad=False
            p.data.copy_(torch.ones_like(p))




