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

#TODO: Add content and style weighting?
#TODO: Figuring out the besting scaling for the log magnitude stft arrays
#      will take some trial and error
#TODO: Accept n_fft on commandline, default to 8192?

# No real rhyme or reason to this
__version__ = "0.0.1"


# Class Definitions
class WavError(Exception):
    pass


class ConvNet(nn.Module):
    def __init__(self, num_frames):
        super(ConvNet, self).__init__()
        self.num_frames = num_frames
        self.conv1 = nn.Conv1d(in_channels=4097, out_channels=self.num_frames,
                               kernel_size=3, stride=1, padding=1)
        # TODO: Add ReLUs, MaxPools, and other Conv Layers
    
    def forward(self, input):
        self.output = self.conv1(input)
        return self.output


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input_data):
        self.loss = F.mse_loss(input_data, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input_data):
        self.G = gram_matrix(input_data)
        self.loss = F.mse_loss(self.G, self.target)
        return input

    # def backward(self, retain_variables=True):
    #     self.loss.backward(retain_graph=True)
    #     return self.loss


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


# TODO
def get_spectrogram_magnitude_tensor(input_wav):
    pass


def plot_spectrogram(stft, out_dir, filename):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max),
                                   y_axis='log', x_axis='time', ax=ax)
    ax.set_title('Spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()
    plt.savefig(join(out_dir, filename))


def phase_reconstruct(input_mag):
    """ Griffin and Lim algorithm for phase reconstruction.

        Note: Assumes that the magnitude information is not on a
              log scale. Make sure to run mag = np.exp(log_mag) - 1
              before passing the data to this function
    """

    # Initialize random phases
    phase = 2 * np.pi * np.random.random_sample(input_mag.shape) - np.pi
    for i in range(500):
        # Compute spectrogram
        spectrogram = input_mag * np.exp(1j*phase)
        # Inverse stft to get signal from mag info and imperfect phase info
        temp_signal = librosa.istft(spectrogram)
        # Recover some meaningful phase info
        phase = np.angle(librosa.stft(temp_signal, 8192))
    
    return phase


def gram_matrix(input_data):
    """ Gram Matrix for Style Loss Module """
    a, b, c, d = input_data.size()
    features = input_data.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d) 


def get_losses(cnn, content_stft, style_stft, device):
    """ Add content loss and style loss layers after convolution step
        
        Note: At this step, content_stft and style_stft should be 
              PyTorch tensors
    """

    cnn = copy.deepcopy(cnn)
    content_losses = []
    style_losses = []

    # TODO: Add layers as the convolutional network gets more complicated

    # Layers of the convolution network to asses content and stlye loss at
    content_layers = ["conv_1"]
    style_layers = ["conv_1"]

    # New network to merge convolution network, content loss network, and
    # style loss network together to calculate content and style losses at 
    # each appropriate convolution network layer
    model = nn.Sequential().to(device)

    layer_level = 1
    for layer in cnn.children():
        if isinstance(layer, nn.Conv1d):
            # Add CNN convolution layer to new model
            name = "conv_" + str(layer_level)
            model.add_module(name, layer)

            if name in content_layers:
                # Add content loss at the current layer
                target = model(content_stft).clone()
                content_loss = ContentLoss(target)
                model.add_module("content_loss" + str(layer_level), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # Add style loss at the current layer
                target = model(style_stft).clone()
                style_loss = StyleLoss(target)
                model.add_module("style_loss" + str(layer_level), style_loss)
                style_losses.append(style_loss)

            # TODO: Add code for ReLUs and MaxPools if they are added 
            #       to the cnn definition
            
            # TODO: Only one layer at the moment so this is useless
            layer_level += 1

    return model, content_losses, style_losses


def get_optimizer(input_data):
    """ Gradient Descent"""
    return optim.LBFGS([input_data.requires_grad_()])


def run_transfer(cnn, content_stft, style_stft, device, num_steps=250):
    """ TODO: If implementing style and content weights at the command line 
              later, pass them to this function as well.
    """
    print("Building model...")

    model, content_losses, style_losses = get_losses(cnn, content_stft, 
                                                     style_stft, device)
    
    optimizer = get_optimizer(content_stft)
    
    print("Optimizing...")

    run = [0]
    while run[0] <= num_steps:

        def closure():
            optimizer.zero_grad()
            model(content_stft)
            content_score = sum([cl.loss for cl in content_losses])
            style_score = sum([sl.loss for sl in style_losses])

            # TODO: Multiply scores by weights
            
            loss = style_score + content_score
            loss.backward()
            
            run[0] += 1

            if run[0] % 10 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    return content_stft


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

    # Separate Short-time Fourier Transforms for left and right channels
    content_wav_l = content_wav[0,:]
    content_wav_r = content_wav[1,:]
    style_wav_l = style_wav[0,:]
    style_wav_r = style_wav[1,:]

    # TODO: Encapsulate the corresponding spectrograms, extract magnitude,
    #       tensor conversion, and tensor reshaping into a separate function
    #       def get_spectrogram_magnitude_tensor(input_wav):

    # Corresponding spectograms 
    c_stft_l = librosa.stft(content_wav_l, n_fft=8192, hop_length=512)
    c_stft_r = librosa.stft(content_wav_r, n_fft=8192, hop_length=512)
    s_stft_l = librosa.stft(style_wav_l, n_fft=8192, hop_length=512)
    s_stft_r = librosa.stft(style_wav_r, n_fft=8192, hop_length=512)

    # Extract log magnitude
    c_stft_l_log_mag = np.log1p(np.abs(c_stft_l))
    c_stft_r_log_mag = np.log1p(np.abs(c_stft_r))
    s_stft_l_log_mag = np.log1p(np.abs(s_stft_l))
    s_stft_r_log_mag = np.log1p(np.abs(s_stft_r))

    # Convert to torch tensors
    c_stft_l_tensor = torch.as_tensor(c_stft_l_log_mag)
    c_stft_r_tensor = torch.as_tensor(c_stft_r_log_mag)
    s_stft_l_tensor = torch.as_tensor(s_stft_l_log_mag)
    s_stft_r_tensor = torch.as_tensor(s_stft_r_log_mag)

    # Reshape tensors to pass to neural net
    c_stft_l_tensor = c_stft_l_tensor.view(-1, c_stft_l.shape[0], c_stft_l.shape[1])
    c_stft_r_tensor = c_stft_r_tensor.view(-1, c_stft_r.shape[0], c_stft_r.shape[1])
    s_stft_l_tensor = s_stft_l_tensor.view(-1, s_stft_l.shape[0], s_stft_l.shape[1])
    s_stft_r_tensor = s_stft_r_tensor.view(-1, s_stft_r.shape[0], s_stft_r.shape[1])

    # Run process on GPU if that's a viable option
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    # Create Convolutional Neural Network
    cnn = ConvNet(num_frames=c_stft_l.shape[1]).to(device)

    # Run the transfer (left only as a test)
    output = run_transfer(cnn, c_stft_l_tensor, s_stft_l_tensor, device, num_steps=250)

    # Plot the content, style, and output stfts
    plot_spectrogram(c_stft_l, out_dir, "content_spectrogram.png")
    plot_spectrogram(s_stft_l, out_dir, "style_spectrogram.png")
    plot_spectrogram(output, out_dir, "output_spectrogram.png")

# Run the script
if __name__ == "__main__":
    main()