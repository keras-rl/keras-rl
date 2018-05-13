from setuptools import setup
from setuptools import find_packages


setup(name='keras-rl',
      version='0.4.0',
      description='Deep Reinforcement Learning for Keras',
      author='Matthias Plappert',
      author_email='matthiasplappert@me.com',
      url='https://github.com/keras-rl/keras-rl',
      download_url='https://github.com/keras-rl/keras-rl/archive/v0.4.0.tar.gz',
      license='MIT',
      install_requires=['keras>=2.0.7'],
      extras_require={
          'gym': ['gym'],
      },
      packages=find_packages())
