# Copyright © 2025 Leadpoet

import re
import os
import codecs
from os import path
from io import open
from setuptools import setup, find_packages

def read_requirements(path):
    """Use the same dependency constraints for source and package installs."""
    with open(path, "r", encoding="utf-8") as f:
        requirements = [line.partition(" #")[0].strip() for line in f]
    return [req for req in requirements if req and not req.startswith("#")]


here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with codecs.open(os.path.join(here, "Leadpoet/__init__.py"), encoding="utf-8") as init_file:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", init_file.read(), re.M)
    if not version_match:
        raise RuntimeError("Unable to find version string in Leadpoet/__init__.py")
    version_string = version_match.group(1)


requirements = read_requirements(path.join(here, "requirements.txt"))

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
    packages=find_packages(include=['Leadpoet', 'Leadpoet.*', 'miner_models', 'miner_models.*', 'neurons', 'neurons.*', 'validator_models', 'validator_models.*', 'leadpoet_audit', 'leadpoet_audit.*', 'gateway', 'gateway.*', 'leadpoet_canonical', 'leadpoet_canonical.*', 'qualification', 'qualification.*', 'leadpoet_verifier', 'leadpoet_verifier.*', 'research_lab', 'research_lab.*', 'leadpoet_observability', 'leadpoet_observability.*', 'lab_arena', 'lab_arena.*', 'validator_tee', 'validator_tee.enclave']) + ['leadpoet_canonical.config'],
    # Ship the existing public mapping as package data, without a second copy
    # in the source tree or a change to deployment's config/ path.
    package_dir={"leadpoet_canonical.config": "config"},
    package_data={
        "leadpoet_verifier": [
            "fixtures/*.json",
            "leadpoet_industry_taxonomy.json",
            "identity/public_suffix_list.dat",
        ],
        "leadpoet_canonical": ["subtensor_events_profile_v2.json"],
        "leadpoet_canonical.config": ["stateful-epoch-cutover-sn71.json"],
        "validator_tee.enclave": [
            "chain_signing_profile_v2.json",
            "chain_signing_profile_test_v2.json",
        ],
        "research_lab": ["engine_program.md", "fixtures/*.json"],
    },
    include_package_data=True,
    python_requires=">=3.11",
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
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Internet :: WWW/HTTP",
        "Topic :: System :: Distributed Computing"
    ],
)
