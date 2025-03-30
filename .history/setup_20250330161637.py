# IN PROGRESS FOR LEADPOET
# The MIT License (MIT)
# Copyright © 2025 Yuma Rao
# Leadpoet
# Copyright © 2025 Leadpoet

import re
import os
import codecs
import pathlib
from os import path
from io import open
from setuptools import setup, find_packages
from pkg_resources import parse_requirements

def read_requirements(path):
    with open(path, "r") as f:
        requirements = f.read().splitlines()
        processed_requirements = []
        for req in requirements:
            # Handle git or VCS links
            if req.startswith("git+") or "@" in req:
                pkg_name = re.search(r"(#egg=)([\w\-_]+)", req)
                if pkg_name:
                    processed_requirements.append(pkg_name.group(2))
                else:
                    continue
            else:
                processed_requirements.append(req)
        return processed_requirements

# Define the project root directory
here = path.abspath(path.dirname(__file__))

# Load the README for the long description
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Load version from template/__init__.py
with codecs.open(os.path.join(here, "template/__init__.py"), encoding="utf-8") as init_file:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", init_file.read(), re.M)
    if not version_match:
        raise RuntimeError("Unable to find version string in template/__init__.py")
    version_string = version_match.group(1)

# Define requirements (manually curated based on LeadPoet needs)
requirements = [
    "bittensor>=6.9.3",  # Core Bittensor library
    "requests>=2.31.0",  # For API calls in get_leads.py and automated_checks.py
    "numpy>=1.24.0",  # For scoring and array operations
    "dnspython>=2.6.1",  # For DNS lookups in automated_checks.py
    "aiohttp>=3.9.5",  # For async HTTP requests if needed in future
    "asyncio>=3.4.3",  # For async operations in neurons and mock
    "pyyaml>=6.0.1",  # For config parsing if extended
    "argparse>=1.4.0",  # For CLI argument parsing
    "pickle-mixin>=1.0.2"  # For caching in automated_checks.py
]

setup(
    name="leadpoet_subnet",  # Unique name for the LeadPoet subnet
    version=version_string,  # Version from template/__init__.py
    description="A Bittensor subnet for decentralized lead generation and validation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Pranav-create/Leadpoet",  # Replace with your GitHub URL
    author="Your Name or Team",  # Replace with your name or team
    author_email="your.email@example.com",  # Replace with your email
    license="MIT",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "leadpoet=neurons.miner:main",  # CLI entry for miners
            "leadpoet-validate=neurons.validator:main",  # CLI entry for validators
            "leadpoet-api=template.api.leadpoet_api:main"  
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Internet :: WWW/HTTP",
        "Topic :: System :: Distributed Computing"
    ],
)