#!/usr/bin/env python

# Neural Timbre Transfer
#
# A Python/PyTorch script to apply the "style" (spectral characteristics/timbre)
# of one audio file to the "content" (melody, etc.) of another audio file.
#
# Raymond Viviano
# August 20th, 2020
# rayviviano@gmail.com

from __future__ import print_function, division
import os, sys, getopt, traceback, functools, warnings
import wave, sndhdr, wavio
import numpy as np 
from os.path import isdir, isfile, abspath, join, basename, splitext, exists
import copy 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# No real rhyme or reason to this
__version__ = "0.0.1"


# Class Definitions
class WavError(Exception):
    pass


# Decorator definitions
def deprecated(func):
    """Decorator to mark deprecated functions. Emit warning on function call"""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)
    return new_func


# Function definitions

def load_wav():
    pass

def compare_wavs():
    pass

def main():
    pass


# Run the script
if __name__ == "__main__":
    main()