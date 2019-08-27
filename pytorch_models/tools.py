import math
import numpy as np
#
# StrideManager
#
class StrideManager(object):
    """

    StrideManager provides two main functions:

        1. Track current output stride of model and provide the correct
           dilation (as property) and (stride) as method so the model can
           have a fixed output stride

        2. Checks the output-stride-state at various points in the model and
           save, the return, features/channels from specific points in the 
           forward pass

    Args:
        output_stride<int>: output_stride
        low_level_output<int|list|str|False>:
            - if truthy append low_level_features/low_level_channels for update_low_level_features
            - preset options exclude the last block
            - options:
                - StrideManager.ALL: after all blocks
                - StrideManager.HALF: at stride = output_stride//2
                - A list containing output strides of interest and/or tag-strings
        drop_array<bool>: 
            - if True and len(low-level-outputs)==1 return value instead of array
    """
    ALL='all'
    HALF='half'


    def __init__(self,
            output_stride=False,
            low_level_output=False,
            drop_array=False):
        self.output_stride=output_stride
        self.low_level_output=low_level_output
        if output_stride:
            self.half_output_stride=int(round(math.sqrt(output_stride)))
            self.scale_factors=self._scale_factors()
        else:
            self.half_output_stride=False
            self.scale_factors=False
        self.drop_array=drop_array
        self.reset()


    def stride(self,stride=2):
        """ returns the correct (dilation dependent) stride
        """
        if self.dilation==1:
            return stride
        else:
            return 1


    def step(self,stride=2,features=None,channels=None,tag=None):
        """ combines increment and update_low_level_features """
        self.increment(stride)
        self.update_low_level_features(
            features=features,
            channels=channels,
            tag=tag)


    def increment(self,stride=2):
        """ updates output_stride_state / dilation 
        """
        self.output_stride_state=self.output_stride_state*stride
        if self.output_stride and (self.output_stride_state>=self.output_stride):
            self.dilation*=stride


    def update_low_level_features(self,features=None,channels=None,tag=None):
        """

        Checks current output_stride_state and tag to see if it is equal to
        or in low_level_output.  If True, appends x and ch to low_level_features and
        low_level_channels respectively.

        Args:
            x<nn.Module>: feature to possibly add to low_level_features
            ch<int>: corresponding channel for low_level_feature
            tag<str|None>: a string indicating which low_level_feature is being tested
        """
        if self.low_level_output:
            if isinstance(self.low_level_output,int):
                is_low_level_feature=self._check_stride(self.low_level_output)
            elif isinstance(self.low_level_output,str):
                if self.low_level_output==StrideManager.ALL:
                    is_low_level_feature=True
                elif self.low_level_output==StrideManager.HALF:
                    is_low_level_feature=self._check_stride(self.half_output_stride)
                else:
                    is_low_level_feature=self.low_level_output==tag
            else:
                state_is_in=any([self._check_stride(s) for s in self.low_level_output])
                if tag:
                    tag_is_in=tag in self.low_level_output
                    is_low_level_feature=state_is_in or tag_is_in
                else:
                    is_low_level_feature=state_is_in
            if is_low_level_feature:
                if features is not None: 
                    self.low_level_features.append(features)
                if channels is not None: 
                    self.low_level_channels.append(channels)


    def features(self):
        """ returns low_level_features 
        * if drop_array=True and if len(low_level_features)==1, return values not arr
        """
        if self.drop_array and (len(self.low_level_features)==1):
            return self.low_level_features[0]
        else:
            return self.low_level_features[::-1]


    def channels(self):
        """ returns low_level_channels 
        * if drop_array=True and if len(low_level_channels)==1, return values not arr
        """
        if self.drop_array and (len(self.low_level_channels)==1):
            return self.low_level_channels[0]
        else:
            return self.low_level_channels[::-1]



    def reset(self):
        """ reset instance
        """
        try:
            del(self.low_level_features)
            del(self.low_level_channels)
        except Exception:
            pass
        self.output_stride_state=1
        self.dilation=1
        self.low_level_features=[]
        self.low_level_channels=[]
        self._existing_strides=[]


    #
    # INTERNAL
    #
    def _scale_factors(self):
        if self.low_level_output==StrideManager.HALF:
            return [self.half_output_stride]
        else:
            test=[isinstance(s,int) for s in self.low_level_output]
            if np.array(test).all():
                os=self.output_stride
                strides=[1]+self.low_level_output
                sfactors=[]
                for s in strides[::-1]:
                    sfactors.append(os/s)
                    os=s
                return sfactors
            else:
                print('WARNING: scale_factors not set. low_level_output must be ints')
                return False


    def _check_stride(self,stride):
        if isinstance(stride,int):
            if stride not in self._existing_strides:
                if stride==self.output_stride_state:
                    self._existing_strides.append(stride)
                    return True



