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
    # Core Bittensor
    "bittensor>=9.10,<10.0.0",
    
    # HTTP and networking
    "requests>=2.31.0",
    "aiohttp>=3.9.5",
    "aiodns>=3.5.0",
    "httpx>=0.28.1",
    
    # Data processing and ML
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    # "torch[cpu]>=2.0.0",  # DEPRECATED: Only needed for unused collusion detection
    # "torch_geometric>=2.4.0",  # DEPRECATED: Only needed for unused collusion detection
    # "pygod>=1.1.0",  # DEPRECATED: Only needed for unused collusion detection
    
    # DNS and validation
    "dnspython>=2.6.1",
    "python-whois>=0.9.5",
    
    # Storage and caching
    "redis>=5.0.0",
    "pickle-mixin>=1.0.2",
    "boto3>=1.40.0",
    "arweave-python-client>=1.0.19",
    
    # Configuration
    "pyyaml>=6.0.1",
    "python-dotenv>=1.0.0",
    "argparse>=1.4.0",
    
    # Validation utilities
    "fuzzywuzzy>=0.18.0",
    "phonenumbers>=8.13.0",
    "disposable-email-domains>=0.0.138",
    "python-Levenshtein>=0.23.0",
    # "ddgs>=6.0.0",  # REMOVED - using ScrapingDog API instead
    
    # Lead Sorcerer dependencies
    "openrouter>=0.0.16",
    "firecrawl>=2.16.0",
    "firecrawl-py>=4.3.6",
    "openai>=2.1.0",
    "portalocker>=2.7.0",
    "publicsuffix2>=2.20191221",
    "jsonschema>=4.25.1",
    
    # Web framework (for gateway)
    "fastapi>=0.110.0",
    "uvicorn>=0.38.0",
    "python-multipart>=0.0.6",
    "pydantic>=2.0.0",
    "starlette>=0.30.0",
    
    # Utilities
    "rich>=13.0.0",
    "click>=8.1.0",
    "typing-extensions>=4.7.0",
    "tenacity>=8.2.0",
    
    # Google Cloud (optional)
    "google-cloud-firestore>=2.11.1",
    "google-auth>=2.17.3",
    "firebase-admin>=6.2.0",
    
    # JWT token management
    "pyjwt>=2.0.0",
    
    # Supabase
    "supabase>=2.0.0",
    
    # TEE Attestation Verification
    "cbor2>=5.4.6",
    "cryptography>=41.0.7",
    
    # gRPC communication
    "grpcio>=1.60.0",
    
    # Additional utilities
    "nest-asyncio>=1.6.0",
    "websockets>=15.0.1",
    
    # Monitoring
    "prometheus_client>=0.19.0",
    "structlog>=23.2.0",
    
    # Geographic normalization (city/state/country standardization)
    "us>=3.0.0",  # US state lookup (offline, no rate limits)
    "geonamescache>=2.0.0",  # 786k+ city name variations (offline, no rate limits)
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
    packages=find_packages(include=['Leadpoet', 'Leadpoet.*', 'miner_models', 'miner_models.*', 'neurons', 'neurons.*', 'validator_models', 'validator_models.*', 'leadpoet_audit', 'leadpoet_audit.*', 'gateway', 'gateway.*']),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "leadpoet=neurons.miner:main",
            "leadpoet-validate=neurons.validator:main",
            "leadpoet-audit=leadpoet_audit.cli:main"
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
