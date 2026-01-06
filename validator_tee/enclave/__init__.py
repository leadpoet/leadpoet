"""
Validator TEE Enclave Module
============================

Files that run INSIDE the Nitro Enclave.
These are packaged into the enclave EIF image.

DO NOT import these from the host - they are designed to run
only inside the hardware-isolated enclave environment.
"""

# No exports - these modules run inside the enclave, not on the host

