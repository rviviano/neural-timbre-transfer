# Neural Timbre Transfer
 CNN Neural Style Transfer for Audio with Python and PyTorch

## Prerequisites 
 [Python3](https://www.python.org/)

 [NumPy](https://numpy.org/)

 [PyTorch](https://pytorch.org/)
 
 [Librosa](https://librosa.org/doc/latest/index.html#)

 [PySoundFile](https://pysoundfile.readthedocs.io/)

 Tested with Python 3.8, NumPy 1.18.5, Torch 1.6.0, Librosa 0.8, Sounfile 0.9, on Windows 10.
 I have not tested if the GPU acceleration works though...

## Installation 
 From the command line (Mac/Linux Terminal or Windows PowerShell).
 
 ```
 python -m pip install git+https://github.com/rviviano/neural-timbre-transfer.git@v0.2.1
 ```
 
## Usage
 ```
 timbre_transfer --c <arg> --s <arg> --o <arg>
 ```
 or
 ```
 python -m neural_timbre_transfer --c <arg> --s <arg> --o <arg>
 ```

### Options

 Mandatory Options:

    --c, --content  
        The wav file to take the content from.
    
    --s, --style    
        The wav file to take the style (timbre/spectral characteristics) from 
        to transfer to the content.

    --o, --out
        Directory to save output wavs to. If the provided path does not exist, 
        then the script will try to create it.

 Optional Options:

    -h             
        Print help message and exit


## Authors
 * **Raymond Viviano** - *Initial Work* - https://github.com/rviviano

## License
 This project is licensed under the MIT License - see the [LICENSE](https://github.com/rviviano/mudpie-sample-generator/blob/master/LICENSE) file for details

## Acknowledgments
 * Thanks to Leon A. Gatys, Alexander S. Ecker, & Matthias Bethge for their [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) paper.
 * Thanks to Eric Grinstein, Ngoc Q. K. Duong, Alexey Ozerov, & Patrick Perez for their [Audio Style Transfer](https://arxiv.org/pdf/1710.11385.pdf) paper.
 * Thanks to Prateek Verma & Julius O. Smith for their [Neural Style Transfer for Audio Spectrograms](https://arxiv.org/pdf/1801.01589.pdf) paper.
 * Thanks to Alexis Jacq for their [PyTorch Style Transfer Tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
 * Thanks to Md Huzaifah for their [Audio Style Transfer Code](https://github.com/muhdhuz/Audio_NeuralStyle/blob/master/neural_style_audio.ipynb) that I would refer to when I was stuck.
