"""
CLI for LeadPoet Community Audit Tool
======================================

Command-line interface for querying and analyzing public transparency log.

Commands:
    leadpoet-audit report <epoch_id>           Generate audit report for epoch
    leadpoet-audit miner <hotkey>              Show miner performance (stub)
    leadpoet-audit compare <epoch1> <epoch2>   Compare epochs (stub)

Author: LeadPoet Team
"""

import click
import json
import sys
from typing import Optional


@click.group()
@click.version_option(version="1.0.0")
def main():
    """
    LeadPoet Community Audit CLI - Query public transparency log
    
    This tool allows anyone to audit and verify lead validation outcomes
    by querying the PUBLIC transparency_log.
    
    Examples:
        leadpoet-audit report 100
        leadpoet-audit report 100 --output epoch_100.json
        leadpoet-audit miner 5GNJqR...
    """
    pass


@main.command()
@click.argument("epoch_id", type=int)
@click.option(
    "--output", "-o", 
    default=None, 
    help="Save report to JSON file (e.g., epoch_100.json)"
)
@click.option(
    "--format", "-f",
    type=click.Choice(["table", "json", "summary"]),
    default="table",
    help="Output format: table (default), json, or summary"
)
def report(epoch_id: int, output: Optional[str], format: str):
    """
    Generate audit report for epoch (queries public transparency_log).
    
    This command downloads CONSENSUS_RESULT events from the public transparency log
    and generates a comprehensive audit report showing:
    
    - Total leads validated and approval rate
    - Top performing miners
    - Most common rejection reasons
    - Rep score distribution
    
    Examples:
        leadpoet-audit report 100
        leadpoet-audit report 100 --output report.json
        leadpoet-audit report 100 --format summary
    """
    try:
        from leadpoet_audit.analyzer import generate_epoch_report
        
        click.echo()
        click.echo(f"📊 Generating audit report for epoch {epoch_id}...")
        click.echo()
        
        # Generate report
        report_data = generate_epoch_report(epoch_id)
        
        # Display based on format
        if format == "json":
            _display_json_report(report_data)
        elif format == "summary":
            _display_summary_report(report_data)
        else:  # table (default)
            _display_table_report(report_data)
        
        # Save to file if requested
        if output:
            _save_report_to_file(report_data, output)
            click.echo()
            click.echo(f"✅ Report saved to {output}")
        
        click.echo()
        
    except Exception as e:
        click.echo()
        click.echo(f"❌ Error generating report: {e}", err=True)
        click.echo()
        sys.exit(1)


def _display_table_report(report: dict):
    """Display report in table format (default, most detailed)."""
    click.echo()
    click.echo("=" * 70)
    click.echo(f"📊 Epoch {report['epoch_id']} Audit Report")
    click.echo("=" * 70)
    click.echo()
    
    # Summary stats
    click.echo("📈 Summary Statistics:")
    click.echo("-" * 70)
    click.echo(f"Total Leads Validated:    {report['total_leads_validated']}")
    click.echo(f"Approved:                 {report['approved_leads']} ({report['approval_rate']}%)")
    click.echo(f"Denied:                   {report['denied_leads']}")
    click.echo(f"Avg Rep Score (All):      {report['avg_rep_score_all']}")
    click.echo(f"Avg Rep Score (Approved): {report['avg_rep_score_approved']}")
    click.echo(f"Unique Miners:            {report['unique_miners']}")
    click.echo(f"Validators Participated:  {report['validator_count']}")
    click.echo()
    
    # Top miners
    if len(report['miner_performance']) > 0:
        click.echo("🏆 Top Performing Miners:")
        click.echo("-" * 70)
        
        # Show top 10 miners
        top_miners = report['miner_performance'].head(10)
        
        # Format as table
        click.echo(f"{'Rank':<6} {'Miner Hotkey':<45} {'Approved':<10} {'Denied':<8} {'Avg Rep':<8}")
        click.echo("-" * 70)
        
        for idx, row in top_miners.iterrows():
            rank = idx + 1
            hotkey = row['miner_hotkey']
            # Truncate hotkey if too long
            if len(hotkey) > 42:
                hotkey = hotkey[:20] + "..." + hotkey[-19:]
            
            click.echo(
                f"{rank:<6} {hotkey:<45} {row['approved_leads']:<10} "
                f"{row['denied_leads']:<8} {row['avg_rep_score']:<8.2f}"
            )
        
        if len(report['miner_performance']) > 10:
            click.echo(f"... and {len(report['miner_performance']) - 10} more miners")
        
        click.echo()
    
    # Rejection reasons
    if len(report['rejection_analysis']) > 0:
        click.echo("❌ Top Rejection Reasons:")
        click.echo("-" * 70)
        
        # Show top 10 reasons
        top_reasons = report['rejection_analysis'].head(10)
        
        # Format as table
        click.echo(f"{'Rejection Reason':<40} {'Count':<8} {'Percentage':<12}")
        click.echo("-" * 70)
        
        for idx, row in top_reasons.iterrows():
            click.echo(
                f"{row['rejection_reason']:<40} {row['count']:<8} {row['percentage']:<12.2f}%"
            )
        
        click.echo()
    else:
        click.echo("✅ No rejections in this epoch!")
        click.echo()
    
    # Rep score distribution
    if 'approval_distribution' in report:
        dist = report['approval_distribution']['rep_score_distribution']
        click.echo("📊 Rep Score Distribution:")
        click.echo("-" * 70)
        
        for bin_range, count in dist.items():
            bar_length = int(count / max(dist.values(), default=1) * 30) if count > 0 else 0
            bar = "█" * bar_length
            click.echo(f"{bin_range:<10} {count:<5} {bar}")
        
        click.echo()
    
    click.echo("=" * 70)


def _display_summary_report(report: dict):
    """Display report in summary format (compact)."""
    click.echo()
    click.echo(f"📊 Epoch {report['epoch_id']} - Summary")
    click.echo()
    click.echo(f"  Leads:        {report['total_leads_validated']}")
    click.echo(f"  Approved:     {report['approved_leads']} ({report['approval_rate']}%)")
    click.echo(f"  Denied:       {report['denied_leads']}")
    click.echo(f"  Avg Rep:      {report['avg_rep_score_all']}")
    click.echo(f"  Miners:       {report['unique_miners']}")
    click.echo(f"  Validators:   {report['validator_count']}")
    click.echo()


def _display_json_report(report: dict):
    """Display report in JSON format."""
    # Convert DataFrames to dicts for JSON serialization
    report_json = {
        "epoch_id": report["epoch_id"],
        "total_leads_validated": report["total_leads_validated"],
        "approved_leads": report["approved_leads"],
        "denied_leads": report["denied_leads"],
        "approval_rate": report["approval_rate"],
        "avg_rep_score_all": report["avg_rep_score_all"],
        "avg_rep_score_approved": report["avg_rep_score_approved"],
        "unique_miners": report["unique_miners"],
        "validator_count": report["validator_count"],
        "miner_performance": report["miner_performance"].to_dict(orient="records"),
        "rejection_analysis": report["rejection_analysis"].to_dict(orient="records"),
        "approval_distribution": report["approval_distribution"]
    }
    
    click.echo(json.dumps(report_json, indent=2))


def _save_report_to_file(report: dict, filepath: str):
    """Save report to JSON file."""
    # Convert DataFrames to dicts for JSON serialization
    report_json = {
        "epoch_id": report["epoch_id"],
        "total_leads_validated": report["total_leads_validated"],
        "approved_leads": report["approved_leads"],
        "denied_leads": report["denied_leads"],
        "approval_rate": report["approval_rate"],
        "avg_rep_score_all": report["avg_rep_score_all"],
        "avg_rep_score_approved": report["avg_rep_score_approved"],
        "unique_miners": report["unique_miners"],
        "validator_count": report["validator_count"],
        "miner_performance": report["miner_performance"].to_dict(orient="records"),
        "rejection_analysis": report["rejection_analysis"].to_dict(orient="records"),
        "approval_distribution": report["approval_distribution"],
        "epoch_assignment": report.get("epoch_assignment"),
        "queue_root": report.get("queue_root")
    }
    
    with open(filepath, 'w') as f:
        json.dump(report_json, f, indent=2)


@main.command()
@click.argument("miner_hotkey")
@click.option("--epoch", "-e", type=int, help="Specific epoch (optional, defaults to latest)")
def miner(miner_hotkey: str, epoch: Optional[int]):
    """
    Show performance for specific miner.
    
    This command queries the transparency log for all leads submitted by a specific miner
    and shows their performance metrics.
    
    Examples:
        leadpoet-audit miner 5GNJqR...
        leadpoet-audit miner 5GNJqR... --epoch 100
    
    Note: This feature is coming soon and will query SUBMISSION and CONSENSUS_RESULT events.
    """
    click.echo()
    click.echo(f"📊 Miner Performance: {miner_hotkey}")
    click.echo("=" * 70)
    click.echo()
    click.echo("⚠️  This feature is coming soon!")
    click.echo()
    click.echo("When implemented, this will show:")
    click.echo("  • Total leads submitted")
    click.echo("  • Approval rate")
    click.echo("  • Average rep score")
    click.echo("  • Performance over time")
    click.echo("  • Most common rejection reasons")
    click.echo()
    click.echo("For now, use: leadpoet-audit report <epoch_id>")
    click.echo("Then search for your miner hotkey in the results.")
    click.echo()


@main.command()
@click.argument("epoch_ids", nargs=-1, type=int, required=True)
@click.option(
    "--output", "-o",
    default=None,
    help="Save comparison to CSV file"
)
def compare(epoch_ids: tuple, output: Optional[str]):
    """
    Compare performance across multiple epochs.
    
    This command generates a comparison table showing trends over time.
    
    Examples:
        leadpoet-audit compare 98 99 100
        leadpoet-audit compare 95 96 97 98 99 100 --output comparison.csv
    
    Note: This feature is coming soon.
    """
    click.echo()
    click.echo(f"📊 Comparing Epochs: {', '.join(map(str, epoch_ids))}")
    click.echo("=" * 70)
    click.echo()
    click.echo("⚠️  This feature is coming soon!")
    click.echo()
    click.echo("When implemented, this will show:")
    click.echo("  • Approval rate trends")
    click.echo("  • Average rep score changes")
    click.echo("  • Number of active miners per epoch")
    click.echo("  • Number of validators per epoch")
    click.echo()
    click.echo("For now, run: leadpoet-audit report <epoch_id> for each epoch.")
    click.echo()


@main.command()
def info():
    """
    Display information about the audit tool and data sources.
    
    Shows:
    - What data is available
    - Where data comes from (public transparency log)
    - How to use the tool
    - Key differences from validator operations
    """
    click.echo()
    click.echo("=" * 70)
    click.echo("📚 LeadPoet Community Audit Tool - Information")
    click.echo("=" * 70)
    click.echo()
    
    click.echo("🎯 Purpose:")
    click.echo("-" * 70)
    click.echo("This tool allows the community to audit and verify lead validation")
    click.echo("outcomes by querying the PUBLIC transparency log.")
    click.echo()
    
    click.echo("📊 Data Source:")
    click.echo("-" * 70)
    click.echo("✅ Public transparency_log table")
    click.echo("✅ CONSENSUS_RESULT events (gateway consensus)")
    click.echo("✅ SUBMISSION events (miner attribution)")
    click.echo("✅ EPOCH_ASSIGNMENT events (assigned leads)")
    click.echo("❌ NO access to private database tables")
    click.echo()
    
    click.echo("⏰ Timing:")
    click.echo("-" * 70)
    click.echo("Run this tool AFTER epoch closes and consensus is computed.")
    click.echo("Consensus is computed at the start of epoch N+1 (after block 360 of epoch N).")
    click.echo()
    
    click.echo("🔑 Key Differences from Validator Operations:")
    click.echo("-" * 70)
    click.echo("Validators:")
    click.echo("  • Calculate weights locally using their OWN decisions")
    click.echo("  • Submit weights to Bittensor chain BEFORE block 360")
    click.echo("  • Bittensor on-chain consensus determines emissions")
    click.echo()
    click.echo("This Audit Tool:")
    click.echo("  • Queries gateway consensus (for transparency only)")
    click.echo("  • Runs AFTER epoch closes")
    click.echo("  • Shows what consensus determined (NOT used for emissions)")
    click.echo()
    
    click.echo("💡 Example Usage:")
    click.echo("-" * 70)
    click.echo("  # Generate report for epoch 100")
    click.echo("  $ leadpoet-audit report 100")
    click.echo()
    click.echo("  # Save report to JSON file")
    click.echo("  $ leadpoet-audit report 100 --output epoch_100.json")
    click.echo()
    click.echo("  # Show summary only")
    click.echo("  $ leadpoet-audit report 100 --format summary")
    click.echo()
    
    click.echo("📚 Learn More:")
    click.echo("-" * 70)
    click.echo("Documentation: See README.md in leadpoet-audit package")
    click.echo("GitHub: https://github.com/leadpoet/leadpoet-audit")
    click.echo()
    click.echo("=" * 70)
    click.echo()


if __name__ == "__main__":
    main()

