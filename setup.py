#!/usr/bin/env python3

from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='rtrain',
    version='0.0.1',
    description='Remote trainer for Keras neural network models.',
    long_description='Remote trainer for Keras neural network models.',
    url='https://github.com/LachlanGunn/rtrain',
    author='Lachlan Gunn',
    author_email='lachlan@twopif.net',
    license='none',
    classifiers=[
        'Development Status :: 1 - Pre-Alpha',
        'Framework :: Flask',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
    ],
    keywords='deeplearning neuralnetworks',
    packages=find_packages(),
    install_requires=['flask', 'keras>=2.0.6', 'tensorflow',  'jsonschema', 'numpy', 'requests', 'tqdm'],
    python_requires='>=3',

    package_data={
        'rtrain': ['schema.sql'],
    },

    entry_points={
        'console_scripts': [
            'rtraind-setup=rtrain.setup_db:main',
            'rtraind=rtrain.server:main',
        ],
    }
)
