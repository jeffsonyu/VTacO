import numpy as np
from collections import defaultdict
from tqdm import tqdm


class BaseInference(object):
    ''' Base trainer class.
    '''
    
    def inference_step(self, *args, **kwargs):
        ''' Performs a training step.
        '''
        raise NotImplementedError

