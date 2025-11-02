#!/usr/bin/env python3
"""
Setup script for CrypTFed Federated Learning Library
"""

from setuptools import setup, find_packages
import os

# Read README if it exists
long_description = "CrypTFed: A Federated Learning Library with Homomorphic Encryption for TensorFlow"
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()

setup(
    name="cryptfed",
    version="1.0.0",
    description="CrypTFed: A Federated Learning Library with Homomorphic Encryption for TensorFlow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="CrypTFed Team",
    author_email="contact@cryptfed.org",
    url="https://github.com/cryptfed/cryptfed",
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "tensorflow>=2.8.0",
        "openfhe-numpy>=0.1.0",
        "tqdm>=4.60.0",
        "pandas>=1.3.0",
        "matplotlib>=3.3.0",
        "psutil>=5.8.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "plotting": [
            "seaborn>=0.11.0",
        ],
        "datasets": [
            "folktables>=1.0.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.900",
        ],
        "all": [
            "seaborn>=0.11.0",
            "folktables>=1.0.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="federated-learning, homomorphic-encryption, privacy-preserving, machine-learning, cryptography",
    project_urls={
        "Bug Reports": "https://github.com/cryptfed/cryptfed/issues",
        "Source": "https://github.com/cryptfed/cryptfed",
        "Documentation": "https://cryptfed.readthedocs.io/",
    },
)