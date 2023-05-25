from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="GeoDeepONet",
    version="0.1.0",
    author="Samuel Burbulla",
    author_email="s.burbulla@appliedai-institute.de",
    description="A Python package for solving partial differential equations on parameterised geometries using DeepONets.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aai-institute/GeoDeepONet",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.1",
        "matplotlib>=3.7.1",
        "tensorboard>=2.13.0",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Intended Audience :: Science/Research",
    ],
)