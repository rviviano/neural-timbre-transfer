#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='neural_timbre_transfer',
      version="0.2.1",
      description='CNN Neural Style Transfer for Audio with Python and PyTorch',
      author='Raymond Viviano',
      author_email='rayviviano@gmail.com',
      packages=find_packages(exclude=["*.test", "*.test.*", "test.*", "test"]),
      py_modules=['neural_timbre_transfer'],
      entry_points = { 
            'console_scripts': [ 
                'timbre_transfer = neural_timbre_transfer:main'
            ] 
        }, 
      license='LICENSE',
    )
