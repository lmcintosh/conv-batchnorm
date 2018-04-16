from setuptools import setup, find_packages
import convbatchnorm

setup(name='convbatchnorm',
      version=convbatchnorm.__version__,
      description='Convolutional batch normalization',
      author='Lane McIntosh',
      author_email='lmcintosh@stanford.edu',
      url='https://github.com/lmcintosh/conv-batchnorm.git',
      install_requires=[i.strip() for i in open("requirements.txt").readlines()],
      long_description='''
          Convolutional batch normalization layer for neural networks
          in Keras.
          ''',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: MacOS :: MacOS X',
          'Topic :: Scientific/Engineering :: Information Analysis'],
      packages=find_packages(),
      )
