"""
LeadPoet Community Audit Tool
==============================

This package provides tools for the community to audit and verify
lead validation outcomes by querying the PUBLIC transparency_log.

**PURPOSE**: Community transparency and verification (NOT for validator weight calculation)

**DATA SOURCE**: Public transparency_log (CONSENSUS_RESULT events)

**NO ACCESS TO**: Private database tables (leads_private, validation_evidence_private)

Usage:
    from leadpoet_audit.downloader import download_consensus_results
    from leadpoet_audit.analyzer import generate_epoch_report
    
    # Generate audit report for epoch
    report = generate_epoch_report(epoch_id=100)
    
    # Or use CLI
    # $ leadpoet-audit report 100

Author: LeadPoet Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "LeadPoet Team"

# Public API (will be available after modules are implemented)
__all__ = [
    "download_consensus_results",
    "download_submission_events",
    "download_epoch_assignment",
    "analyze_miner_performance",
    "analyze_rejection_reasons",
    "generate_epoch_report"
]

# Import modules when available (graceful degradation during development)
try:
    from leadpoet_audit.downloader import (
        download_consensus_results,
        download_submission_events,
        download_epoch_assignment
    )
except ImportError:
    pass  # Modules not yet implemented

try:
    from leadpoet_audit.analyzer import (
        analyze_miner_performance,
        analyze_rejection_reasons,
        generate_epoch_report
    )
except ImportError:
    pass  # Modules not yet implemented

