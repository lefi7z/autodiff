from autodiff import __version__
from setuptools import setup

opts = dict(
    name='autodiff', 
    setup_requires=['numpy>=1.0'],
    version=__version__,
    packages=['autodiff'],
    description='Automatic differentiation package.',
)

if __name__ == "__main__":
    setup(**opts)

