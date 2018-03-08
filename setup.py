from setuptools import setup
from setuptools import find_packages


setup(name='keras-rl',
      version='0.4.0',
      description='Deep Reinforcement Learning for Keras',
      author='Matthias Plappert',
      author_email='matthiasplappert@me.com',
      url='https://github.com/matthiasplappert/keras-rl',
      download_url='https://github.com/matthiasplappert/keras-rl/archive/v0.4.0.tar.gz',
      license='MIT',
      install_requires=['keras>=2.0.7,<=2.1.3'],
      extras_require={
          'gym': ['gym'],
      },
      packages=find_packages())
