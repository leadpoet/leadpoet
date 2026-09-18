# Copyright © 2025 Leadpoet

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

requirements = read_requirements(path.join(here, "requirements.txt"))

setup(
    name="leadpoet_subnet",  
    version="0.0.0",
    description="A Bittensor subnet for decentralized lead generation and validation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leadpoet/leadpoet",  
    author="Leadpoet",  
    author_email="hello@leadpoet.com",  
    license="AGPL-3.0-only",
    packages=find_packages(include=['Leadpoet', 'Leadpoet.*', 'gateway', 'gateway.*', 'leadpoet_canonical', 'leadpoet_canonical.*', 'qualification', 'qualification.*', 'leadpoet_verifier', 'leadpoet_verifier.*', 'leadpoet_observability', 'leadpoet_observability.*', 'lab_arena', 'lab_arena.*', 'neurons', 'validator_tee', 'validator_tee.enclave']) + ['leadpoet_canonical.config'],
    # Ship the existing public mapping as package data, without a second copy
    # in the source tree or a change to deployment's config/ path.
    package_dir={"leadpoet_canonical.config": "config"},
    package_data={
        "leadpoet_verifier": [
            "leadpoet_industry_taxonomy.json",
            "identity/public_suffix_list.dat",
        ],
        "leadpoet_canonical": [
            "subtensor_events_profile_v2.json",
            "subtensor_events_profile_spec*_v2.json",
        ],
        "leadpoet_canonical.config": ["stateful-epoch-cutover-sn71.json"],
        "validator_tee.enclave": [
            "chain_signing_profile_v2.json",
            "chain_signing_profile_test_v2.json",
        ],
    },
    include_package_data=True,
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "leadpoet=lab_arena.miner_cli:main",
            "leadpoet-validate=lab_arena.validator:main",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Internet :: WWW/HTTP",
        "Topic :: System :: Distributed Computing"
    ],
)
