# Do not eagerly import validator-only helpers here. Scoring runtimes import
# ``Leadpoet.utils.utils_lead_extraction`` without Bittensor installed, while
# validator callers import ``Leadpoet.utils.config`` and the other submodules
# explicitly when they need them.
