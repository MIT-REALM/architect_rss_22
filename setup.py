#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="architect",
    version="0.0.0",
    description="Automated, robust co-design",
    author="Charles Dawson",
    author_email="cbd@mit.edu",
    url="https://github.com/dawsonc/architect_env",
    install_requires=[],
    package_data={"architect": ["py.typed"]},
    packages=find_packages(),
)
