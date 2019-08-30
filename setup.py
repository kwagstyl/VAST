from setuptools import setup, find_packages

setup(name='vast',
      version='0.1',
      packages=find_packages(),
      install_requires=['nibabel','h5py']
     )


