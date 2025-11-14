"""
Data Analyzer for LeadPoet Community Audit Tool
================================================

This module provides functions to analyze CONSENSUS_RESULT events from the public transparency_log.

**DATA SOURCE**: Public transparency_log (CONSENSUS_RESULT events)
**PURPOSE**: Community transparency - shows what consensus determined (NOT used for emissions)

All functions work with PUBLIC data downloaded via the downloader module.

Author: LeadPoet Team
"""

import pandas as pd
from typing import Dict, List
from collections import defaultdict, Counter


def analyze_miner_performance(
    consensus_results: List[Dict],
    submissions: List[Dict]
) -> pd.DataFrame:
    """
    Analyze miner performance based on gateway CONSENSUS_RESULT events.
    
    This shows what the gateway consensus determined (for transparency/auditing only).
    NOT used for actual emissions - validators calculate weights locally.
    
    Args:
        consensus_results: List of CONSENSUS_RESULT payloads from transparency_log
        submissions: List of SUBMISSION events (to map lead_id -> miner_hotkey)
    
    Returns:
        DataFrame with columns: [miner_hotkey, approved_leads, denied_leads, 
                                  avg_rep_score, total_rep_score]
        Sorted by total_rep_score descending (best performers first)
    
    Example:
        >>> from leadpoet_audit.downloader import download_consensus_results, download_submission_events
        >>> consensus_results = download_consensus_results(100)
        >>> submissions = download_submission_events()
        >>> df = analyze_miner_performance(consensus_results, submissions)
        >>> print(df.head())
        miner_hotkey              approved_leads  denied_leads  avg_rep_score  total_rep_score
        5GNJqR...                             12             2          85.30          1023.60
        5FHneW...                              8             3          82.10           656.80
        ...
    """
    # Build lead_id -> miner_hotkey mapping
    lead_to_miner = {}
    for sub in submissions:
        lead_to_miner[sub["lead_id"]] = sub["miner_hotkey"]
    
    # Aggregate per miner
    miner_stats = defaultdict(lambda: {
        "approved_leads": 0,
        "denied_leads": 0,
        "total_rep_score": 0.0,
        "count": 0
    })
    
    for result in consensus_results:
        lead_id = result["lead_id"]
        miner_hotkey = lead_to_miner.get(lead_id, "unknown")
        
        final_decision = result["final_decision"]
        final_rep_score = result.get("final_rep_score", 0.0)
        
        if final_decision == "approve":
            miner_stats[miner_hotkey]["approved_leads"] += 1
            miner_stats[miner_hotkey]["total_rep_score"] += final_rep_score
        else:
            miner_stats[miner_hotkey]["denied_leads"] += 1
        
        miner_stats[miner_hotkey]["count"] += 1
    
    # Build DataFrame
    rows = []
    for miner_hotkey, stats in miner_stats.items():
        avg_rep_score = stats["total_rep_score"] / stats["count"] if stats["count"] > 0 else 0.0
        
        rows.append({
            "miner_hotkey": miner_hotkey,
            "approved_leads": stats["approved_leads"],
            "denied_leads": stats["denied_leads"],
            "total_leads": stats["count"],
            "avg_rep_score": round(avg_rep_score, 2),
            "total_rep_score": round(stats["total_rep_score"], 2)
        })
    
    df = pd.DataFrame(rows)
    
    # Sort by total_rep_score descending (best performers first)
    if len(df) > 0:
        df = df.sort_values("total_rep_score", ascending=False)
        df = df.reset_index(drop=True)
    
    return df


def analyze_rejection_reasons(consensus_results: List[Dict]) -> pd.DataFrame:
    """
    Analyze most common rejection reasons from CONSENSUS_RESULT events.
    
    This shows why leads were denied according to gateway consensus.
    Helps identify common quality issues in submitted leads.
    
    Args:
        consensus_results: List of CONSENSUS_RESULT payloads
    
    Returns:
        DataFrame with columns: [rejection_reason, count, percentage]
        Sorted by count descending (most common first)
    
    Example:
        >>> from leadpoet_audit.downloader import download_consensus_results
        >>> consensus_results = download_consensus_results(100)
        >>> df = analyze_rejection_reasons(consensus_results)
        >>> print(df)
        rejection_reason              count  percentage
        email invalid                     8       40.00
        LinkedIn match failed             6       30.00
        catch-all email                   4       20.00
        company name mismatch             2       10.00
    """
    # Count rejection reasons (excluding "pass" for approved leads)
    reasons = []
    for result in consensus_results:
        if result["final_decision"] == "deny":
            reason = result.get("primary_rejection_reason", "unknown")
            # Exclude "pass" (should not appear for denied leads, but just in case)
            if reason and reason.lower() != "pass":
                reasons.append(reason)
    
    if not reasons:
        # Return empty DataFrame if no rejections
        return pd.DataFrame(columns=["rejection_reason", "count", "percentage"])
    
    reason_counts = Counter(reasons)
    total = sum(reason_counts.values())
    
    rows = []
    for reason, count in reason_counts.most_common():
        percentage = (count / total * 100) if total > 0 else 0.0
        rows.append({
            "rejection_reason": reason,
            "count": count,
            "percentage": round(percentage, 2)
        })
    
    df = pd.DataFrame(rows)
    return df


def analyze_approval_distribution(consensus_results: List[Dict]) -> Dict:
    """
    Analyze distribution of approval decisions and rep_scores.
    
    Args:
        consensus_results: List of CONSENSUS_RESULT payloads
    
    Returns:
        Dictionary with approval statistics:
        {
            'total_leads': int,
            'approved_count': int,
            'denied_count': int,
            'approval_rate': float,
            'avg_rep_score_all': float,
            'avg_rep_score_approved': float,
            'rep_score_distribution': Dict[str, int]  # Bins: 0-20, 20-40, etc.
        }
    
    Example:
        >>> stats = analyze_approval_distribution(consensus_results)
        >>> print(f"Approval rate: {stats['approval_rate']}%")
        Approval rate: 76.0%
    """
    if not consensus_results:
        return {
            'total_leads': 0,
            'approved_count': 0,
            'denied_count': 0,
            'approval_rate': 0.0,
            'avg_rep_score_all': 0.0,
            'avg_rep_score_approved': 0.0,
            'rep_score_distribution': {}
        }
    
    total_leads = len(consensus_results)
    approved = [r for r in consensus_results if r["final_decision"] == "approve"]
    denied = [r for r in consensus_results if r["final_decision"] == "deny"]
    
    approved_count = len(approved)
    denied_count = len(denied)
    approval_rate = (approved_count / total_leads * 100) if total_leads > 0 else 0.0
    
    # Calculate average rep scores
    all_rep_scores = [r.get("final_rep_score", 0.0) for r in consensus_results]
    approved_rep_scores = [r.get("final_rep_score", 0.0) for r in approved]
    
    avg_rep_score_all = sum(all_rep_scores) / len(all_rep_scores) if all_rep_scores else 0.0
    avg_rep_score_approved = sum(approved_rep_scores) / len(approved_rep_scores) if approved_rep_scores else 0.0
    
    # Rep score distribution (bins: 0-20, 20-40, 40-60, 60-80, 80-100)
    distribution = {
        '0-20': 0,
        '20-40': 0,
        '40-60': 0,
        '60-80': 0,
        '80-100': 0
    }
    
    for score in all_rep_scores:
        if score < 20:
            distribution['0-20'] += 1
        elif score < 40:
            distribution['20-40'] += 1
        elif score < 60:
            distribution['40-60'] += 1
        elif score < 80:
            distribution['60-80'] += 1
        else:
            distribution['80-100'] += 1
    
    return {
        'total_leads': total_leads,
        'approved_count': approved_count,
        'denied_count': denied_count,
        'approval_rate': round(approval_rate, 2),
        'avg_rep_score_all': round(avg_rep_score_all, 2),
        'avg_rep_score_approved': round(avg_rep_score_approved, 2),
        'rep_score_distribution': distribution
    }


def generate_epoch_report(epoch_id: int) -> Dict:
    """
    Generate comprehensive audit report for epoch (AFTER consensus computed).
    
    This is the main function that orchestrates all analysis.
    Downloads data from public transparency_log and generates complete report.
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        Report dictionary with all stats and DataFrames:
        {
            'epoch_id': int,
            'total_leads_validated': int,
            'approved_leads': int,
            'denied_leads': int,
            'approval_rate': float,
            'avg_rep_score_all': float,
            'avg_rep_score_approved': float,
            'unique_miners': int,
            'validator_count': int,
            'miner_performance': pd.DataFrame,
            'rejection_analysis': pd.DataFrame,
            'approval_distribution': Dict,
            'epoch_assignment': Dict,
            'queue_root': Dict
        }
    
    Example:
        >>> from leadpoet_audit.analyzer import generate_epoch_report
        >>> report = generate_epoch_report(epoch_id=100)
        >>> 
        >>> print(f"Epoch {report['epoch_id']}:")
        >>> print(f"  Total Leads: {report['total_leads_validated']}")
        >>> print(f"  Approval Rate: {report['approval_rate']}%")
        >>> print(f"  Unique Miners: {report['unique_miners']}")
        >>> 
        >>> # View top miners
        >>> print(report['miner_performance'].head(10))
        >>> 
        >>> # View rejection reasons
        >>> print(report['rejection_analysis'])
    """
    from leadpoet_audit.downloader import (
        download_consensus_results,
        download_submission_events,
        download_epoch_assignment,
        download_queue_root
    )
    
    print(f"\n📊 Generating audit report for epoch {epoch_id}...")
    print("=" * 70)
    
    # Download PUBLIC data from transparency_log
    consensus_results = download_consensus_results(epoch_id)
    submissions = download_submission_events()
    assignment = download_epoch_assignment(epoch_id)
    queue_root = download_queue_root(epoch_id)
    
    print()
    print("🔍 Analyzing data...")
    
    # Analyze miner performance
    miner_performance = analyze_miner_performance(consensus_results, submissions)
    
    # Analyze rejection reasons
    rejection_analysis = analyze_rejection_reasons(consensus_results)
    
    # Analyze approval distribution
    approval_distribution = analyze_approval_distribution(consensus_results)
    
    # Extract validator count (from first consensus result if available)
    validator_count = 0
    if consensus_results:
        validator_count = consensus_results[0].get("validator_count", 0)
    
    # Build comprehensive report (epoch-specific metrics grouped separately)
    report = {
        "miner_performance": miner_performance,
        "rejection_analysis": rejection_analysis,
        "approval_distribution": approval_distribution,
        "epoch_metrics": {
            "epoch_id": epoch_id,
            "leads_validated_this_epoch": len(consensus_results),
            "leads_assigned_this_epoch": assignment['lead_count'] if assignment else 0,
            "validator_count": validator_count,
            "queue_state": {
                "pending_leads": queue_root['pending_count'] if queue_root else 0,
                "queue_merkle_root": queue_root['queue_root'] if queue_root else None
            },
            "assignment": {
                "lead_ids": assignment['lead_ids'] if assignment else [],
                "validator_count": assignment['validator_count'] if assignment else 0
            }
        }
    }
    
    print()
    print("=" * 70)
    print(f"✅ Report generated successfully for epoch {epoch_id}")
    print()
    
    return report


def compare_epochs(epoch_ids: List[int]) -> pd.DataFrame:
    """
    Compare performance across multiple epochs.
    
    Utility function to analyze trends over time.
    
    Args:
        epoch_ids: List of epoch numbers to compare
    
    Returns:
        DataFrame with epoch comparison stats
    
    Example:
        >>> df = compare_epochs([98, 99, 100])
        >>> print(df)
        epoch_id  total_leads  approval_rate  avg_rep_score  unique_miners
        98               50          72.0           78.5               7
        99               50          76.0           79.2               8
        100              50          80.0           81.1               9
    """
    rows = []
    
    for epoch_id in epoch_ids:
        try:
            report = generate_epoch_report(epoch_id)
            epoch_metrics = report['epoch_metrics']
            approval_dist = report['approval_distribution']
            rows.append({
                'epoch_id': epoch_metrics['epoch_id'],
                'total_leads': epoch_metrics['leads_validated_this_epoch'],
                'approved': approval_dist['approved_count'],
                'denied': approval_dist['denied_count'],
                'approval_rate': approval_dist['approval_rate'],
                'avg_rep_score': approval_dist['avg_rep_score_all'],
                'unique_miners': len(report['miner_performance']),
                'validators': epoch_metrics['validator_count']
            })
        except Exception as e:
            print(f"⚠️  Error processing epoch {epoch_id}: {e}")
            continue
    
    df = pd.DataFrame(rows)
    return df

