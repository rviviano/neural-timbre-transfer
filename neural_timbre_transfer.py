#!/usr/bin/env python

# Neural Timbre Transfer
#
# A Python/PyTorch script to apply the "style" (spectral characteristics/timbre)
# of one audio file to the "content" (melody, etc.) of another audio file.
#
# Raymond Viviano
# August 20th, 2020
# rayviviano@gmail.com

# TODO: Consider adding monophonic and polyphonic pitch detection of the 
# input wavs, then pitchbend/timestretch the "style" audio wav to match
# the "content" audio wav.


from __future__ import print_function, division
import os, sys, getopt, traceback, functools, warnings
import librosa, librosa.display
import numpy as np 
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


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4097, out_channels=4097,
                               kernel_size=3, stride=1, padding=1)

    
    def forward(self, x):
        pass


class ContentLoss(nn.Module):

    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


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

    Usage: python neural_timbre_transfer --content <arg> --style <arg>  
                                         --out <arg> -h

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
                content_path = arg

        if opt == "--s" or opt == "--style":
            if (arg is not None) and (isfile(abspath(arg))):
                style_path = arg

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


def load_wav(wav_filepath):
    """Load wav data into np array, resample to 44k"""
    wv, _ = librosa.load(wav_filepath, sr = 44100, mono=False)                                
    return  wv


def compare_wavs_length(c_wv, s_wv):
    """         
        If the style wav is longer than the content wav, truncate the style
        wav to be the same length as the content wav. If the style wav is 
        shorter than the content wav, exit in error. Looping the style wav 
        to match the length of content wav might add an unwanted audible 
        artifact at the looping point.

        Inputs: c_wv : "content" wav
                s_wv : "sytle" wav
    """

    # If the style wav is shorter than the content wav, exit in error
    if s_wv.shape[0] < c_wv.shape[0]:
        raise WavError("The style wav must be as long or longer than the content wav")

    # If the style wav is longer, truncate it to match the content length
    elif s_wv.shape[0] > c_wv.shape[0]:
        print("Truncating style wav to match content wav length")
        s_wv = np.copy(s_wv[:c_wv.shape[0],:])

    return s_wv


def plot_spectrogram(stft):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max),
                                   y_axis='log', x_axis='time', ax=ax)
    ax.set_title('Spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()


def gram_matrix(input):
    """ Gram Matrix for Style Loss Module """
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d) 


def get_losses(cnn, content_stft, style_stft):
    """ Add content loss and style loss layers after convolution step"""
    pass





def main():
    # Get input wavs and output directory from commandline
    content_path, style_path, out_dir = process_options() 

    # Load the input wav files
    content_wav = load_wav(content_path)
    style_wav = load_wav(style_path)

    # Ensure that the wavs are mutable
    content_wav.flags.writeable = True
    style_wav.flags.writeable = True

    # Check compatibility of wav lengths
    style_wav = compare_wavs_length(content_wav, style_wav)

    # Define an output wav 
    output_wav = np.copy(content_wav)

    # Separate Short-time Fourier Transforms for left and right channels
    content_wav_l = content_wav[0,:]
    content_wav_r = content_wav[1,:]
    style_wav_l = style_wav[0,:]
    style_wav_r = style_wav[1,:]

    # Corresponding spectograms 
    c_stft_l = librosa.stft(content_wav_l, n_fft=8192, hop_length=512)
    c_stft_r = librosa.stft(content_wav_r, n_fft=8192, hop_length=512)
    s_stft_l = librosa.stft(style_wav_l, n_fft=8192, hop_length=512)
    s_stft_r = librosa.stft(style_wav_r, n_fft=8192, hop_length=512)

    # Convert to torch tensors
    c_stft_l_tensor = torch.as_tensor(c_stft_l)
    c_stft_r_tensor = torch.as_tensor(c_stft_r)
    s_stft_l_tensor = torch.as_tensor(s_stft_l)
    s_stft_r_tensor = torch.as_tensor(s_stft_r)
    # Run the process on the GPU if that's a viable option
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    # Hyperparameters
    learning_rate = .01

    # Create Convolutional Neural Network
    model = ConvNet().to(device)


# Run the script
if __name__ == "__main__":
    main()