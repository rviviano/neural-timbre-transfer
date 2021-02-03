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
import soundfile as sf
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
#TODO: Accept hop_length specification on command line?
#TODO: Add more verbose print msgs throughout script
#TODO: Implement a 2d-conv option, also define a gram matrix function specific 
#      to the 2d-conv case
#TODO: Add maxpooling and additional convolution when implementing the 2d case
#TODO: 2d-convolution might be better at detecting pitch bends
#TODO: set 'center' to false for all stfts and istfts
#TODO: figure out if pulsing/banding has to do with hop_length specs, loss_function specs,
#      or if it's just related to the test audio
#TODO: Try additional test audio

__version__ = "0.1.0"


# Class Definitions
class WavError(Exception):
    pass


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4097, out_channels=4097,
                               kernel_size=3, stride=1, padding=1)
        # TODO: Add ReLUs, MaxPools, and other Conv Layers
    
    def forward(self, input):
        self.output = self.conv1(input)
        return self.output


class ContentLoss(nn.Module):
    def __init__(self, target, content_weight):
        super(ContentLoss, self).__init__()
        self.content_weight = content_weight
        self.target = target.detach() * self.content_weight
        
    def forward(self, input_data):
        self.loss = F.mse_loss(input_data * self.content_weight, self.target)
        self.output = input_data
        return self.output

    def backward(self):
        self.loss.backward(retain_graph=True)
        return self.loss


class StyleLoss(nn.Module):
    def __init__(self, target_feature, style_weight):
        super(StyleLoss, self).__init__()
        self.style_weight = style_weight
        self.target = target_feature.detach() * self.style_weight

    def forward(self, input_data):
        self.output = input_data.detach()
        self.G = gram_matrix(input_data).mul_(self.style_weight)
        self.loss = F.mse_loss(self.G, self.target, size_average=False)
        return self.output

    def backward(self):
        self.loss.backward(retain_graph=True)
        return self.loss


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


def get_spectrogram_magnitude_tensor(in_wav):
    # Corresponding spectogram
    D = librosa.stft(in_wav, n_fft=8192, hop_length=512)
    
    # Extract log magnitude
    log_mag = np.log1p(np.abs(D))

    # Convert to torch tensors
    out_tensor = torch.as_tensor(log_mag)
    out_tensor.contiguous()
    
    # Reshape tensors to pass to neural net
    out_tensor = out_tensor.view(-1, D.shape[0], D.shape[1])
        
    return out_tensor

# TODO: Add argument for title
def plot_spectrogram(stft, out_dir, filename):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(stft), ref=np.max),
                                   y_axis='log', x_axis='time', ax=ax)
    ax.set_title('Spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    # plt.show()
    plt.savefig(join(out_dir, filename))


def phase_reconstruct(input_mag):
    """ Griffin and Lim algorithm for phase reconstruction.

        Note: Assumes that the magnitude information is not on a
              log scale. Make sure to run mag = np.exp(log_mag) - 1
              before passing the data to this function
    """

    print("Reconstructing phase information...")
    # Initialize random phases
    phase = 2 * np.pi * np.random.random_sample(input_mag.shape) - np.pi
    for i in range(500):
        # Compute spectrogram
        spectrogram = input_mag * np.exp(1j*phase)
        # Inverse stft to get signal from mag info and imperfect phase info
        temp_signal = librosa.istft(spectrogram, hop_length=512)
        # Recover some meaningful phase info
        phase = np.angle(librosa.stft(temp_signal, hop_length=512, n_fft=8192))

        if i % 25 == 0:
            print(str(round(i*.2, 2)) + "% complete")
    
    return phase


def gram_matrix(input_data):
    """ Gram Matrix for Style Loss Module

        a = batch size (1)
        b = number of feature maps
        c*d = number of features in a feature map

        Note: This specification is specific to 1d convolution
    
    """
    _, b, cd = input_data.size() # Only passing 3dim tensors
    features = input_data.view(b, cd) 
    G = torch.mm(features, features.t())
    return G.div(b * cd) 


def get_model_and_losses(cnn, content_tensor, style_tensor, content_weight, 
                         style_weight, device):
    """ Add transparent content and style loss layers after convolution step """

    cnn = copy.deepcopy(cnn)
    content_losses = []
    style_losses = []

    # TODO: Add layers as the convolutional network gets more complicated

    # Layers of the convolution network to asses content and style loss at
    content_layers = ["conv_1"]
    style_layers = ["conv_1"]

    # New network to merge convolution network, content loss network, and
    # style loss network together to calculate content and style losses at 
    # each appropriate convolution network layer. Added layers are transparent.
    model = nn.Sequential().to(device)

    layer_level = 1
    for layer in cnn.children():
        if isinstance(layer, nn.Conv1d):
            # Add CNN convolution layer to new model
            name = "conv_" + str(layer_level)
            model.add_module(name, layer)

            if name in content_layers:
                # Add content loss at the current layer
                target = model(content_tensor).detach()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss" + str(layer_level), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # Add style loss at the current layer
                target = model(style_tensor).detach()
                target_gram = gram_matrix(target)
                style_loss = StyleLoss(target_gram, style_weight)
                model.add_module("style_loss" + str(layer_level), style_loss)
                style_losses.append(style_loss)

            # TODO: Add code for ReLUs and MaxPools if they are added 
            #       to the cnn definition
            
            # TODO: Only one layer at the moment so this is useless
            layer_level += 1

    return model, content_losses, style_losses


def get_optimizer(content_tensor):
    """ Gradient Descent"""
    content_param = nn.Parameter(content_tensor.data)
    optimizer = optim.LBFGS([content_param.requires_grad_()])
    return content_param, optimizer

# TODO: Figure out why run_transfer takes 10 to 20 more steps than specified...
def run_transfer(cnn, content_tensor, style_tensor, device, content_weight=1, 
                 style_weight=10, num_steps=2000):
    """ TODO: If implementing style and content weighting at the command line 
              later, pass them to this function as well.
    """
    print("Building model...")

    model, content_losses, style_losses = get_model_and_losses(cnn, content_tensor, style_tensor, content_weight, style_weight, device)
    
    content_param, optimizer = get_optimizer(content_tensor.contiguous())

    print("Optimizing...")

    run = [0]
    while run[0] <= num_steps:

        # TODO: Consider impact of input clamping
        def closure():
            optimizer.zero_grad()
            model(content_param)

            content_score = sum([cl.backward() for cl in content_losses])
            style_score = sum([sl.backward() for sl in style_losses])

            loss = style_score + content_score

            run[0] += 1

            if run[0] % 10 == 0:
                print("Run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return loss

        optimizer.step(closure)

    return content_param.data


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

    # Get spectrograms, extract magnitude, convert to tensors, reshape
    c_stft_l_tensor = get_spectrogram_magnitude_tensor(content_wav_l)
    c_stft_r_tensor = get_spectrogram_magnitude_tensor(content_wav_r)
    s_stft_l_tensor = get_spectrogram_magnitude_tensor(style_wav_l)
    s_stft_r_tensor = get_spectrogram_magnitude_tensor(style_wav_r)

    # Run process on GPU if that's a viable option
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    # Create Convolutional Neural Network
    cnn = ConvNet().to(device)

    # Run the transfer on left signals
    print("Running timbre transfer on left channel...")
    output_l = run_transfer(cnn, c_stft_l_tensor, s_stft_l_tensor, device, 
                            content_weight=1, style_weight=50, num_steps=1750)
    

    # Run the transfer on right signals
    print("Running timbre transfer on right channel...")
    output_r = run_transfer(cnn, c_stft_r_tensor, s_stft_r_tensor, device, 
                            content_weight=1, style_weight=50, num_steps=1750)
    
    output_l = output_l.cpu()
    output_r = output_r.cpu()
    output_l = output_l.squeeze(0)
    output_r = output_r.squeeze(0)

    # Inverse log
    output_l = output_l.numpy()
    output_r = output_r.numpy()

    output_mag_l = np.exp(output_l) - 1
    output_mag_r = np.exp(output_r) - 1

    # Recover phase
    print("Recovering phase for left channel...")
    phase_l = phase_reconstruct(output_mag_l)

    print("Recovering phase for right channel...")
    phase_r = phase_reconstruct(output_mag_r)

    # Combine magnitude and phase information
    output_stft_l = output_mag_l * np.exp(1j*phase_l)
    output_stft_r = output_mag_r * np.exp(1j*phase_r)

    # Plot the content, style, and output stfts for the left channel
    plot_spectrogram(librosa.stft(content_wav_l, n_fft=8192, hop_length=512), 
                     out_dir, "content_spectrogram-left.png")

    plot_spectrogram(librosa.stft(style_wav_l, n_fft=8192, hop_length=512), 
                     out_dir, "style_spectrogram-left.png")

    plot_spectrogram(output_stft_l, out_dir, "output_spectrogram-left.png")

    # Plot the content, style, and output stfts for the right channel
    plot_spectrogram(librosa.stft(content_wav_r, n_fft=8192, hop_length=512), 
                     out_dir, "content_spectrogram-right.png")

    plot_spectrogram(librosa.stft(style_wav_r, n_fft=8192, hop_length=512), 
                     out_dir, "style_spectrogram-right.png")

    plot_spectrogram(output_stft_r, out_dir, "output_spectrogram-right.png")

    # Convert stfts back to signals
    out_signal_l = librosa.istft(output_stft_l, hop_length=512)
    out_signal_r = librosa.istft(output_stft_r, hop_length=512)

    # Combine left and right channels
    out_wav = np.zeros((2, out_signal_l.shape[0]))
    out_wav[0,:] = out_signal_l
    out_wav[1,:] = out_signal_r

    # Save output wav file (Soundfile expects frames by channels)
    sf.write(join(out_dir, "timbre-transfer-output.wav"), out_wav.T, 44100)

# Run the script
if __name__ == "__main__":
    main()