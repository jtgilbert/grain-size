#!usr/bin/env python

from setuptools import setup
import re

# https://packaging.python.org/discussions/install-requires-vs-requirements/
install_requires = [
    'geopandas>=0.11', 'numpy>=1.21', 'pandas>=1.4', 'scipy>=1.9', 'scikit-learn>=1.1.2',
]

with open("README.md", "rb") as f:
    long_descr = f.read().decode("utf-8")

version = re.search(
    '^__version__\\s*=\\s*"(.*)"',
    open('__version__.py').read(),
    re.M
).group(1)

setup(name='grain_size_distributions',
      version=version,
      author='Jordan Gilbert',
      license='MIT',
      python_requires='>=3.8',
      long_description=long_descr,
      author_email='jtgilbert89@gmail.com',
      install_requires=install_requires,
      zip_safe=False,
      entry_points={
          "console_scripts": ['grain_size:main']
      },
      url='https://github.com/jtgilbert/grain-size',
      packages=[
          'grain_size'
      ])
