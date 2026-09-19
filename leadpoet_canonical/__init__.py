"""Shared deterministic hashing, signing, chain, and reward primitives.

Normal Arena validators construct weights from accepted Arena reward state.
The protected signer verifies the same arithmetic and constrains the final
transaction. Generic gateway qualification and artifact services use their
own shared verification primitives from this package.
"""

# Version of the canonical module
__version__ = "1.0.0"
