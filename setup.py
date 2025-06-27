"""
Setup script for CLIP++ package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="clip-plus-plus",
    version="1.0.0",
    author="CLIP++ Team",
    author_email="contact@example.com",
    description="A PyTorch implementation of CLIP with enhanced prompt learning capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/clip-plus-plus",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.7.1",
        "torchvision>=0.8.0",
        "Pillow>=8.0.0",
        "numpy>=1.19.0",
        "tqdm>=4.50.0",
        "requests>=2.25.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.800",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 