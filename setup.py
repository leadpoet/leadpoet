# Copyright © 2025 Leadpoet

import re
import os
import codecs
from os import path
from io import open
from setuptools import setup, find_packages

def read_requirements(path):
    with open(path, "r") as f:
        requirements = f.read().splitlines()
        processed_requirements = []
        for req in requirements:
            if req.startswith("git+") or "@" in req:
                pkg_name = re.search(r"(#egg=)([\w\-_]+)", req)
                if pkg_name:
                    processed_requirements.append(pkg_name.group(2))
                else:
                    continue
            else:
                processed_requirements.append(req)
        return processed_requirements


here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with codecs.open(os.path.join(here, "Leadpoet/__init__.py"), encoding="utf-8") as init_file:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", init_file.read(), re.M)
    if not version_match:
        raise RuntimeError("Unable to find version string in Leadpoet/__init__.py")
    version_string = version_match.group(1)


requirements = [
    "bittensor>=6.9.3", 
    "requests>=2.31.0",
    "numpy>=1.24.0",  
    "dnspython>=2.6.1",
    "aiohttp>=3.9.5",  
    "asyncio>=3.4.3",  
    "pyyaml>=6.0.1",  
    "argparse>=1.4.0",  
    "pickle-mixin>=1.0.2",
    "pygod>=1.1.0",
    "torch>=2.0.0",
    "torch_geometric>=2.4.0",
    "python-whois>=0.9.5",
    "aiodns>=3.5.0",
    "httpx>=0.28.1",
    "openai>=2.1.0",
    "jsonschema>=4.25.1",
    "firecrawl-py>=4.3.6",
    "disposable-email-domains>=0.0.138",
    "fuzzywuzzy>=0.18.0",
    # Lead Sorcerer dependencies
    "openrouter>=1.0.0",
    "firecrawl>=2.16.0",
    "phonenumbers>=8.13.0",
    "portalocker>=2.7.0",
    "publicsuffix2>=2.20191221",
    "python-dotenv>=1.0.0",
    # gRPC communication
    "grpcio>=1.60.0",
    # JWT token management
    "pyjwt>=2.0.0",
    # Supabase client
    "supabase>=2.0.0"
]

setup(
    name="leadpoet_subnet",  
    version=version_string,
    description="A Bittensor subnet for decentralized lead generation and validation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leadpoet/leadpoet",  
    author="Leadpoet",  
    author_email="hello@leadpoet.com",  
    license="MIT",
    packages=find_packages(include=['Leadpoet', 'Leadpoet.*', 'miner_models', 'miner_models.*', 'neurons', 'neurons.*', 'validator_models', 'validator_models.*']),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "leadpoet=neurons.miner:main",
            "leadpoet-validate=neurons.validator:main"
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
