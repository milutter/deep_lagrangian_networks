from setuptools import setup

setup(
    name='deep_lagrangian_networks',
    version='0.1',
    url='https://github.com/milutter/deep_lagrangian_networks.git',

    description='This package provides an implementation of a Deep Lagrangian Networks.',

    author='Michael Lutter',
    author_email='michael@robot-learning.de',

    packages=['deep_lagrangian_networks', ],

    classifiers=['Development Status :: 3 - Alpha'], install_requires=['matplotlib',
                                                                       'numpy',
                                                                       'torch',
                                                                       'dill'])