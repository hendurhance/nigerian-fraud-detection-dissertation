#!/usr/bin/env python
"""Setup script for NIBSS Fraud Detection System."""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nigerian-fraud-detection-dissertation",
    version="1.0.0",
    author="Josiah Endurance Owolabi",
    author_email="hendurhance.dev@gmail.com",
    description="A comprehensive fraud detection system for Nigerian banking transactions using machine learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hendurhance/nigerian-fraud-detection-dissertation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
        "License :: OSI Approved :: CC BY 4.0 License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "isort>=5.10.0",
            "mypy>=0.991",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "jupyterlab>=3.4.0",
            "notebook>=6.4.0",
        ],
        "visualization": [
            "plotly>=5.10.0",
            "seaborn>=0.11.0",
            "matplotlib>=3.5.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "nibss-generate-data=nibss_fraud_dataset_generator:main",
        ],
    },
    package_data={
        "": ["config/*.yaml", "data/external/*.csv"],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="fraud detection, machine learning, Nigerian banking, NIBSS, fintech",
    project_urls={
        "Bug Reports": "https://github.com/hendurhance/nibss-fraud-detection/issues",
        "Source": "https://github.com/hendurhance/nibss-fraud-detection",
    },
)