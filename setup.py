"""Installation script for the 'wbc_mjlab' python package."""

from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mjlab==1.2.0",
]

# Installation operation
setup(
    name="wbc_mjlab",
    packages=["src"],
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
)
