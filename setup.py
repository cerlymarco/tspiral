import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.1.1'
PACKAGE_NAME = 'tspiral'
AUTHOR = 'Marco Cerliani'
AUTHOR_EMAIL = 'cerlymarco@gmail.com'
URL = 'https://github.com/cerlymarco/tspiral'

LICENSE = 'MIT'
DESCRIPTION = 'A python package for time series forecasting with scikit-learn estimators.'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
    'scikit-learn>=0.24.2',
    'numpy'
]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      python_requires='>=3',
      packages=find_packages()
      )