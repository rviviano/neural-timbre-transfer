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
import wave, sndhdr, wavio # TODO: Shift to using pysoundfile and librosa
import numpy as np 
from scipy import signal # Shift to using librosa?
from os.path import isdir, isfile, abspath, join, basename, splitext, exists
import copy 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# import matplotlib
# matplotlib.use('agg')
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
def process_options():
    """
        Process command line arguments for file inputs and output. 
        Also provide a help option. If specific options that could
        usefully influence the style transfer come up, implement them.
    """

    # Define usage
    usage = """

    Usage: python mudpie_sample_generator --in <arg> --out <arg> --pre <arg> 
                                          --num <arg> -h

    Mandatory Options:

        --c, --content  The wav file to take the content from.
        
        --s, --style    The wav file to take the style (timbre/spectral 
                        characteristics) from to transfer to the content.

        --o, --out      Directory to save output wavs to. If the provided path 
                        does not exist, then the script will try to create it.

    Optional Options:

        -h             Print this usage message and exit

    """

    # Get commandline options and arguments
    opts, _ = getopt.getopt(sys.argv[1:], "h:", ["content=", "style=", "out="])

    # Set variables to defaults
    content_path, style_path, out_dir = None, None, None

    for opt, arg in opts:
        # Mandatory arguments
        if opt == "--c" or opt == "--content":
            if (arg is not None) and (isfile(abspath(arg))):
                content_path = check_input_arg(arg)

        if opt == "--s" or opt == "--style":
            if (arg is not None) and (isfile(abspath(arg))):
                style_path = check_input_arg(arg)

        if opt == "--o" or opt == "--out": 
            if arg is not None:
                out_dir = arg
                # If the specified out dir does not exist, try to make it
                if not isdir(abspath(out_dir)): 
                    try:
                        os.makedirs(abspath(out_dir))
                    except:
                        traceback.print_exc(file=sys.stdout)
                        print(os.linesep + 'Cannot create output directory.')
                        sys.exit(1)

        # Optional options
        if opt == "-h":
            print(usage)
            sys.exit(0)


    # Make sure that arguments exist for all mandatory options
    if None in [content_path, style_path, out_dir]:
        print(os.linesep + 'Errors detected with mandatory options')
        print(usage)
        sys.exit(1)

    # Return options for audio processing
    return content_path, style_path, out_dir


def check_input_arg(arg):
    """ Check that file supplied on the command line is a 16 or 24 bit wav """
    try:
        wv_hdr = sndhdr.what(abspath(arg))
        if wv_hdr is not None:
            input_path = arg
            return input_path
        else: 
            raise WavError("You must supply 16- or 24-bit int wavs")
    except WavError:
        raise
    except:
        traceback.print_exc(file=sys.stdout)
        print(os.linesep + 'Unexpected input error.')
        sys.exit(1)

@deprecated
def load_wav(wav_filepath):
    """Load wav data into np array and also return important wav parameters"""
    wv = wavio.read(wav_filepath)
    wav_data = wv.data 
    framerate = wv.rate
    samplewidth = wv.sampwidth                                   
    return  wav_data, framerate, samplewidth


def compare_wavs(c_wv, c_rt, c_wd, s_wv, s_rt, s_wd,):
    """ 
        Ensure that the input wavs have the same bit-depth and sample rate.
        If they do not, exit in error. Also check that number of channels
        are comparable and that there are no more than two. This script is
        not going to support surround sound audio at this time.

        NOTE: I know the sample rates must match because that ensures that
        the wavs have the same Nyquist frequency and that the bandwidths are
        the same. However, I'm not yet sure that the wavs need to have the same
        bitdepth. I'm going to implement this constraint anyway; but I could
        also probably just normalize the sample amplitudes. I could probably 
        implement a samplerate conversion too. But that is a TODO for another day.
        
        If the style wav is longer than the content wav, truncate the style
        wav to be the same length as the content wav. If the style wav is 
        shorter than the content wav, exit in error. Looping the style wav 
        to match the length of content wav might add an unwanted audible 
        artifact at the looping point.
    """

    print("Comparing wav parameters")
    print("-"*80)
    print("Content Sample Rate: " + str(c_rt))
    print("Style Sample Rate:   " + str(s_rt))
    print("-"*80)
    print("Content Sample Width: " + str(c_wd))
    print("Style Sample Width:   " + str(s_wd))
    print("-"*80)
    print("Content Frames: " + str(c_wv.shape[0]))
    print("Style Frames:   " + str(s_wv.shape[0]))
    print("-"*80)
    
    if c_rt != s_rt:
        raise WavError("Sample rates of input wavs must match")

    if c_wd != s_wd:
        raise WavError("Sample bit-depths of input wavs must match")

    # If the style wav is shorter than the content wav, exit in error
    if s_wv.shape[0] < c_wv.shape[0]:
        raise WavError("The style wav must be as long or longer than the content wav")

    # If the style wav is longer, truncate it to match the content length
    elif s_wv.shape[0] > c_wv.shape[0]:
        print("Truncating style wav to match content wav length")
        s_wv = np.copy(s_wv[:c_wv.shape[0],:])

    # TODO: Check that the number of channels in the style wav matches the
    # number of channels in the content wav.

    return s_wv


def main():
    # Run the process on the GPU if that's a viable option
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    # Get input wavs and output directory from commandline
    content_path, style_path, out_dir = process_options() 

    # Load the input wav files
    content_wav, content_frm_rate, content_smp_width = load_wav(content_path)
    style_wav, style_frm_rate, style_smp_width = load_wav(style_path)

    # Make sure that the wavs are mutable, probably only necessary for style wav
    content_wav.flags.writeable = True
    style_wav.flags.writeable = True

    # Check that the wav files have the sample parameters
    style_wav = compare_wavs(content_wav, content_frm_rate, content_smp_width, 
                             style_wav, style_frm_rate, style_smp_width)

    # Define an output wav 
    output_wav = np.copy(content_wav)

    # Lets work with just the left channel for now
    content_wav_l = content_wav[:,0]
    style_wav_l = style_wav[:,0]

    c_f, c_t, c_stft = signal.stft(content_wav_l, fs=content_frm_rate, nperseg=8192)

    print()

    plt.pcolormesh(c_t, c_f, np.abs(c_stft), shading='gouraud')
    # plt.yscale('log')
    # plt.ylim(ymax=20000)
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


# Run the script
if __name__ == "__main__":
    main()