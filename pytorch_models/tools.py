import pandas as pd

#
# StrideManager
#
class StrideManager(object):
    COLUMNS=[
        'tag',
        'stride_state',
        'dilation',
        'channels',
        'low_level_feature']
    """
    Usage:
        
        Make a first through: 
            sm=StrideManager(...)
            sm.update(ch1,...)
            sm.update(ch2,...)
            ...

        Restart and step through, checking if low-level-feature or interest

            x=...
            sm.start()
            feats=[]
            x=layer_1(x)
            if sm.step():
                feats.append(x)
            x=layer_2(x)
            if sm.step():
                feats.append(x)
            ...
    """
    def __init__(self,
                 output_stride=False,
                 indices=[],
                 tags=None,
                 stride_states=None):
        self.output_stride=output_stride
        self.tags=tags or []
        self.stride_states=stride_states or []
        if tags or stride_states:
            self.indices=[]
        else:
            self.indices=indices
        self.reset()

                
    def stride(self,stride=2):
        """ returns the correct (dilation dependent) stride
        """
        if self.dilation==1:
            return stride
        else:
            return 1


    def update(self,channels,stride=2,tag=None):
        """ 
        * updates output_stride_state/dilation 
        * append steps
        """
        self.index+=1
        if not tag: tag=f'step_{self.index}'
        stride_state=self.output_stride_state*stride
        if not self._stop:
            self.output_stride_state=stride_state
        if self.output_stride and (stride_state>=self.output_stride):
            self._stop=True
            self.dilation*=stride
        if (tag in self.tags) or self._first_state(stride_state):
            self.indices.append(self.index)
        self.steps.append({
            'tag': tag,
            'channels': channels,
            'stride_state': self.output_stride_state,
            'dilation': self.dilation,
            'low_level_feature': self.index in self.indices
        })

        
    def start(self):
        """
        """
        self.index=-1
        self.output_stride_state=1
        self.dilation=1

        
    def step(self):
        """ 
        * increment index 
        * set properties
        * check if index in indices
        """
        self.index+=1
        row=self.describe().iloc[self.index]
        self.tag=row.tag
        self.channels=row.channels
        self.stride_state=row.stride_state
        self.dilation=row.dilation
        self.low_level_feature=row.low_level_feature
        return self.low_level_feature
        
        
    def describe(self):
        """ get dataframe describing the steps
        """
        if self._df is None:
            self._df=pd.DataFrame(self.steps)
            self._df=self._df[StrideManager.COLUMNS]
        return self._df

    
    def channels(self):
        """ output channels for low-level-features in reverse order
        """
        if not self._channels:
            self._channels=[self.steps[i]['channels'] for i in self.indices[::-1]]
        return self._channels
    
    
    def scale_factors(self):
        """ scale factors for low-level-features in reverse order
        """        
        if not self._scales:
            os=self.steps[-1]['stride_state']
            self._scales=[]
            for index in self.indices[::-1]:
                ss=self.steps[index]['stride_state']
                self._scales.append(os/ss)
                os=ss
            self._scales.append(os)
        return self._scales
            
        
    def __len__(self):
        """ number of steps """
        return len(self.steps)
    

    def reset(self):
        self.start()
        self.steps=[]
        self._existing_states=[]
        self._stop=False
        self._scales=False        
        self._channels=False        
        self._df=None



    #
    # INTERNAL
    #
    def _first_state(self,stride_state):
        if (stride_state in self.stride_states):
            if stride_state not in self._existing_states:
                self._existing_states.append(stride_state)
                return True



