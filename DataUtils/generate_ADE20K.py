# -*- coding: utf-8 -*-
from __future__ import print_function
import os
from Utils import *
import warnings
warnings.filterwarnings('ignore')
devices = 'cuda' if torch.cuda.is_available() else 'cpu'

# selected class
# television receiver, bus,
# car, oven, person and bicycle in our evaluation.


def generate(path):
    all_data = ''
