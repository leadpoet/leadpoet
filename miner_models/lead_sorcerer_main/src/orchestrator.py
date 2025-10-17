"""
Orchestrator for Lead Sorcerer.

This orchestrator coordinates the three main tools (Domain, Crawl, Enrich) to process
leads through the complete pipeline. It handles data flow, persistence, exports,
and respects all BRD constraints.

Authoritative specifications: BRD §410-439
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from src.common import (
    PermitManager,
    setup_logging,
    validate_provider_config,
    load_costs_config,
    round4,
    now_z,
    normalize_role_priority,
    build_error,
    ErrorCode,
)
from src.domain import DomainTool
from src.crawl import CrawlTool

# ============================================================================
# Constants and Configuration
# ============================================================================

INTERNAL_VERSION = "1.0.0"
DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_CONCURRENT_REQUESTS = 3

# ============================================================================
# Orchestrator Class
# ============================================================================


class LeadSorcererOrchestrator:
    """
    Main orchestrator that coordinates Domain → Crawl → Enrich pipeline.

    Responsibilities:
    - Load icp_config.json and resolve data directory
    - Call tools in sequence with proper filtering
    - Persist results to JSONL files
    - Handle exports when enabled
    - Manage concurrency and batch processing
    """

    def __init__(self, config_path: str, batch_size: int = DEFAULT_BATCH_SIZE):
        """
        Initialize the orchestrator.

        Args:
            config_path: Path to icp_config.json
            batch_size: Number of leads to process per batch
        """
        self.config_path = config_path
        self.batch_size = batch_size

        # Load configuration first
        self.icp_config = self._load_icp_config()

        # Resolve data directory
        self.data_dir = self._resolve_data_dir()

        # Establish a single shared log file for this run
        logs_dir = Path(self.data_dir) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
        shared_log_file = str(logs_dir / f"run_{run_timestamp}.log")
        # Expose to child tools via env so all use the same file
        os.environ["LEADSORCERER_LOG_FILE"] = shared_log_file

        # Setup logging after data_dir is resolved, using the shared log file
        self.logger = setup_logging("orchestrator",
                                    data_dir=self.data_dir,
                                    log_file_path=shared_log_file)

        # Initialize permit manager for global concurrency control
        max_concurrent = self.icp_config.get("concurrency", {}).get(
            "max_concurrent_requests", DEFAULT_MAX_CONCURRENT_REQUESTS)
        self.permit_manager = PermitManager(max_concurrent)

        # Initialize tools based on lead generation mode
        self._initialize_tools()

        # Track metrics
        self.start_time = None
        self.total_errors = []
        self.total_unknown_errors = 0

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass

    def _initialize_tools(self) -> None:
        """Initialize tools for traditional lead generation mode."""
        # Traditional mode: Initialize domain and crawl tools only
        self.domain_tool = DomainTool(self.icp_config, self.data_dir)
        self.crawl_tool = CrawlTool(self.data_dir)

    def _detect_lead_generation_mode(self) -> str:
        """
        Detect the lead generation mode from ICP configuration.

        Returns:
            Lead generation mode: "traditional" or "specific_urls"
        """
        mode = self.icp_config.get("lead_generation_mode", "traditional")

        if mode not in ["traditional", "specific_urls"]:
            self.logger.warning(
                f"Invalid lead_generation_mode: {mode}, defaulting to 'traditional'"
            )
            mode = "traditional"

        self.logger.info(f"🔍 Lead generation mode detected: {mode}")
        return mode

    def _load_icp_config(self) -> Dict[str, Any]:
        """Load and validate ICP configuration."""
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)

            # Validate required fields
            required_fields = [
                "name",
                "icp_text",
                "queries",
                "required_fields",
                "threshold",
            ]
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required field: {field}")

            # Normalize role_priority configuration
            if "role_priority" in config:
                config["role_priority"] = normalize_role_priority(
                    config["role_priority"])

            return config
        except Exception as e:
            self.logger.error(f"Failed to load ICP config: {e}")
            raise

    def _resolve_data_dir(self) -> str:
        """Resolve data directory with precedence."""
        # Check environment variable first
        env_data_dir = os.environ.get("LEADPOET_DATA_DIR")
        if env_data_dir:
            return env_data_dir

        # Default to ./data
        return "./data"

    def _validate_provider_config(self) -> None:
        """Validate that all required providers are configured."""
        try:
            costs_config = load_costs_config()
            required_providers = [
                "gse",
                "openrouter",
                "firecrawl",
            ]
            validate_provider_config(required_providers, costs_config)
        except Exception as e:
            self.logger.error(f"Provider configuration validation failed: {e}")
            raise

    def _validate_testing_flags(self) -> None:
        """Validate testing configuration flags."""
        testing = self.icp_config.get("testing", {})

        # Validate process_rejected flag
        if not isinstance(testing.get("process_rejected", False), bool):
            raise ValueError("testing.process_rejected must be a boolean")

        # Validate process_low_crawl flag
        if not isinstance(testing.get("process_low_crawl", False), bool):
            raise ValueError("testing.process_low_crawl must be a boolean")

        self.logger.info(
            f"Testing flags: process_rejected={testing.get('process_rejected', False)}, process_low_crawl={testing.get('process_low_crawl', False)}"
        )

    def _validate_refresh_policy(self) -> None:
        """Validate refresh policy configuration."""
        refresh_policy = self.icp_config.get("refresh_policy", {})

        # Validate domain_ttl_days override
        domain_ttl_days = refresh_policy.get("domain_ttl_days")
        if domain_ttl_days is not None and not isinstance(
                domain_ttl_days, int):
            raise ValueError(
                "refresh_policy.domain_ttl_days must be an integer")

        # Validate search.max_pages override
        search_config = self.icp_config.get("search", {})
        max_pages = search_config.get("max_pages")
        if max_pages is not None and not isinstance(max_pages, int):
            raise ValueError("search.max_pages must be an integer")

        # Validate default_country
        default_country = self.icp_config.get("default_country")
        if default_country and not isinstance(default_country, str):
            raise ValueError("default_country must be a string")

        # Validate role_priority configuration
        role_priority = self.icp_config.get("role_priority", {})
        if not isinstance(role_priority, dict):
            raise ValueError("role_priority must be a dictionary")

        self.logger.info(
            f"Refresh policy validated: domain_ttl_days={domain_ttl_days}, max_pages={max_pages}, default_country={default_country}"
        )

    def should_bypass_domain_discovery(self) -> bool:
        """
        Check if we should bypass domain discovery and go directly to crawling specific URLs.
        This saves API costs and enables rapid iteration on known targets.

        Returns:
            True if we should bypass domain discovery and crawl specific URLs directly
        """
        specific_urls = self.icp_config.get("specific_urls", [])
        self.icp_config.get("bypass_domain_discovery", False)

        # Bypass domain discovery if we have specific URLs
        # This prioritizes specific targets for rapid iteration and cost optimization
        has_specific_urls = len(specific_urls) > 0

        # COST OPTIMIZATION: Always bypass when specific URLs are provided
        # This saves API costs by skipping search queries
        should_bypass = has_specific_urls

        if should_bypass:
            self.logger.info(
                f"🚀 BYPASSING DOMAIN DISCOVERY: {len(specific_urls)} specific URLs provided"
            )
            self.logger.info(
                "💰 Cost optimization: Skipping search queries, going directly to crawl"
            )
            self.logger.info(
                f"📊 Specific URLs: {specific_urls[:3]}{'...' if len(specific_urls) > 3 else ''}"
            )

        return should_bypass

    def create_lead_records_from_specific_urls(self) -> List[Dict[str, Any]]:
        """
        Create lead records directly from specific URLs, bypassing domain discovery.
        This enables rapid iteration and cost optimization.

        Returns:
            List of lead records ready for crawling
        """
        specific_urls = self.icp_config.get("specific_urls", [])
        lead_records = []

        for i, url in enumerate(specific_urls):
            # Extract domain from URL
            from urllib.parse import urlparse

            parsed = urlparse(url)
            domain = parsed.netloc

            # Create lead record with complete schema compliance
            lead_record = {
                "lead_id": f"specific-url-{i + 1:03d}",
                "domain": domain,
                "status": "scored",  # Ready for crawl - skip domain scoring
                "status_history":
                [],  # Initialize status history for enrich tool
                "company": {
                    "name":
                    f"{self.icp_config.get('company_name_template', 'Company from {domain}').format(domain=domain)}",
                    "description": f"Direct crawl from specific URL: {url}",
                },
                "contacts": [],  # Will be populated by enrich tool
                "provenance": {
                    "source": "specific_urls",
                    "specific_url": url,
                    "bypassed_domain_discovery": True,
                    "created_at": now_z(),
                    # Add missing schema fields for specific URL mode
                    "queries": [],  # No search queries used
                    "discovery_evidence": [],  # No discovery performed
                    "scored_at": now_z(),  # Set to creation time
                    "crawled_at": None,  # Will be set by crawl tool
                    "enriched_at": None,  # Will be set by enrich tool
                    "next_revisit_at": None,  # Will be set by enrich tool
                    "tool_versions": {
                        "domain": "bypassed",
                        "crawl": None,  # Will be set by crawl tool
                        "enrich": None,  # Will be set by enrich tool
                    },
                    "cache": {
                        "hit": False,
                        "key": None
                    },  # No domain cache used
                    "evidence_paths": {},  # Will be populated by tools
                },
                "icp": {
                    "pre_score":
                    1.0,  # Assume high value since it's a specific target
                    "pre_pass":
                    True,
                    "pre_reason":
                    "Specific URL target - bypassing domain scoring",
                    "threshold":
                    self.icp_config.get("threshold", 0.1),
                    "filtering_strict":
                    self.icp_config.get("filtering_strict", False),
                    # Add missing schema fields for specific URL mode
                    "pre_flags": [],  # No pre-analysis flags
                    "scoring_meta": {
                        "method": "bypass",
                        "model": "none",
                        "prompt_fingerprint": "specific-url-bypass",
                        "temperature": 0.0,
                    },
                    "crawl_score":
                    None,  # Will be set by crawl tool
                    "crawl_reason":
                    None,  # Will be set by crawl tool
                },
                "audit": [],  # Will be populated by tools
                "cost": {
                    "domain_usd": 0.0,
                    "crawl_usd": 0.0,
                    "enrich_usd": 0.0,
                    "total_usd": 0.0,
                },
            }

            lead_records.append(lead_record)

        self.logger.info(
            f"📋 Created {len(lead_records)} lead records from specific URLs")
        self.logger.info(
            "💰 Cost optimization: Bypassed domain discovery phase")

        return lead_records

    async def run_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete lead generation pipeline.

        Returns:
            Pipeline results with metrics and summary
        """
        self.start_time = time.time()
        self.logger.info(
            f"Starting Lead Sorcerer pipeline with config: {self.icp_config['name']}"
        )

        try:
            # Validate provider configuration
            self._validate_provider_config()

            # Validate testing flags and refresh policy
            self._validate_testing_flags()
            self._validate_refresh_policy()

            # Run traditional pipeline (Domain → Crawl only)
            return await self._run_traditional_pipeline()

        except Exception as e:
            print("error in orch")
            self.logger.error(f"Pipeline failed: {e}")
            return {
                "success":
                False,
                "errors":
                [build_error(ErrorCode.UNKNOWN, exc=e, tool="orchestrator")],
                "metrics":
                self._build_error_metrics(),
            }

    async def _run_traditional_pipeline(self) -> Dict[str, Any]:
        """
        Run the traditional Domain → Crawl pipeline.

        Returns:
            Pipeline results with metrics and summary
        """
        self.logger.info(
            "📊 Running traditional lead generation pipeline (Domain → Crawl only)"
        )

        # Check if we should bypass domain discovery for specific URLs
        if self.should_bypass_domain_discovery():
            # BYPASS MODE: Create lead records directly from specific URLs
            self.logger.info(
                "🚀 BYPASS MODE: Creating lead records from specific URLs")
            lead_records = self.create_lead_records_from_specific_urls()

            # Create mock domain result for consistency
            domain_result = {
                "data": {
                    "lead_records": lead_records
                },
                "errors": [],
                "metrics": {
                    "count_in": 0,
                    "count_out": len(lead_records),
                    "duration_ms": 0,
                    "cost_usd": {
                        "domain": 0.0,
                        "crawl": 0.0,
                        "enrich": 0.0,
                        "total": 0.0,
                    },
                },
            }

            # All records from specific URLs are considered "passing"
            passing_leads = lead_records
            self.logger.info(
                f"💰 Cost savings: Skipped domain discovery for {len(passing_leads)} specific URL targets"
            )
        else:
            # NORMAL MODE: Run domain discovery as usual
            self.logger.info("📊 NORMAL MODE: Running domain discovery")
            domain_result = await self.domain_tool.run()

            if domain_result.get("errors"):
                self.total_errors.extend(domain_result["errors"])

            lead_records = domain_result.get("data",
                                             {}).get("lead_records", [])
            self.logger.info(
                f"Domain tool processed {len(lead_records)} leads")

            # Filter to passing leads for next stage
            passing_leads = [
                r for r in lead_records if r.get("icp", {}).get("pre_pass")
            ]
            self.logger.info(f"Filtered to {len(passing_leads)} passing leads")

        # Persist domain results (pass + rejects) - works for both modes
        self._persist_domain_results(lead_records)

        # Step 2: Crawl tool - extract company and contact data
        crawl_result = None

        if passing_leads:
            self.logger.info("Running Crawl tool...")
            crawl_payload = {
                "lead_records": passing_leads,
                "icp_config": self.icp_config,
            }
            crawl_result = await self.crawl_tool.run(crawl_payload)

            if crawl_result.get("errors"):
                self.total_errors.extend(crawl_result["errors"])

            crawled_leads = crawl_result.get("data",
                                             {}).get("lead_records", [])
            self.logger.info(
                f"Crawl tool processed {len(crawled_leads)} leads")

            # Add detailed logging for pipeline flow
            self.logger.info(
                f"🔍 Pipeline Debug: Found {len(crawled_leads)} crawled leads")
            status_counts = {}
            for lead in crawled_leads:
                status = lead.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
            self.logger.info(
                f"📊 Crawled leads status breakdown: {status_counts}")

            # Update lead_records with crawled data
            lead_records = crawled_leads
        else:
            self.logger.info("No passing leads to crawl")

        # Handle exports if enabled
        if self.icp_config.get("exports", {}).get("enabled", False):
            self._export_leads(lead_records)

        # Calculate final metrics
        final_metrics = self._calculate_final_metrics(domain_result,
                                                      crawl_result, None)

        # Log summary
        self._log_pipeline_summary(final_metrics)

        return {
            "success": True,
            "metrics": final_metrics,
            "total_errors": len(self.total_errors),
            "unknown_error_count": self.total_unknown_errors,
            "errors": self.total_errors,
        }

    def _persist_domain_results(self, lead_records: List[Dict[str,
                                                              Any]]) -> None:
        """Persist domain results to JSONL files."""
        try:
            # Ensure data directory exists
            Path(self.data_dir).mkdir(parents=True, exist_ok=True)

            # Write all results (pass + rejects)
            all_file = Path(self.data_dir) / "domain_all.jsonl"
            with open(all_file, "a", encoding="utf-8") as f:
                for record in lead_records:
                    f.write(json.dumps(record) + "\n")

            # Write only passing results
            pass_file = Path(self.data_dir) / "domain_pass.jsonl"
            passing_records = [
                r for r in lead_records if r.get("icp", {}).get("pre_pass")
            ]
            with open(pass_file, "a", encoding="utf-8") as f:
                for record in passing_records:
                    f.write(json.dumps(record) + "\n")

            self.logger.info(
                f"Persisted {len(lead_records)} total leads, {len(passing_records)} passing leads"
            )

            # Check if rotation is needed
            self._check_and_rotate_sinks(all_file, pass_file)

        except Exception as e:
            self.logger.error(f"Failed to persist domain results: {e}")

    def _check_and_rotate_sinks(self, all_file: Path, pass_file: Path) -> None:
        """
        Check if sinks need rotation (>10 MB or >30 days) and rotate if needed.

        Uses copy-on-write compaction to ensure atomic pointer swaps.
        """
        try:
            current_time = datetime.utcnow()

            # Check domain_all.jsonl
            if all_file.exists():
                self._rotate_sink_if_needed(all_file, current_time,
                                            "domain_all")

            # Check domain_pass.jsonl
            if pass_file.exists():
                self._rotate_sink_if_needed(pass_file, current_time,
                                            "domain_pass")

        except Exception as e:
            self.logger.error(f"Failed to check/rotate sinks: {e}")

    def _rotate_sink_if_needed(self, sink_file: Path, current_time: datetime,
                               sink_name: str) -> None:
        """
        Rotate a sink file if it exceeds size or time limits.

        Args:
            sink_file: Path to the sink file
            current_time: Current UTC time
            sink_name: Name of the sink for logging
        """
        try:
            # Check file size (>10 MB)
            file_size_mb = sink_file.stat().st_size / (1024 * 1024)
            size_exceeded = file_size_mb > 10

            # Check file age (>30 days)
            file_mtime = datetime.fromtimestamp(sink_file.stat().st_mtime)
            age_days = (current_time - file_mtime).days
            age_exceeded = age_days > 30

            if size_exceeded or age_exceeded:
                self.logger.info(
                    f"Rotating {sink_name}: size={file_size_mb:.1f}MB, age={age_days} days"
                )
                self._rotate_sink_with_compaction(sink_file, sink_name)

        except Exception as e:
            self.logger.error(f"Failed to check rotation for {sink_name}: {e}")

    def _rotate_sink_with_compaction(self, sink_file: Path,
                                     sink_name: str) -> None:
        """
        Rotate sink using copy-on-write compaction for atomic pointer swap.

        Args:
            sink_file: Path to the sink file
            sink_name: Name of the sink for logging
        """
        try:
            # Create timestamped archive name
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
            archive_name = f"{sink_name}_{timestamp}.jsonl"
            archive_path = sink_file.parent / archive_name

            # Copy current content to archive (copy-on-write)
            if sink_file.exists():
                with (
                        open(sink_file, "r", encoding="utf-8") as src,
                        open(archive_path, "w", encoding="utf-8") as dst,
                ):
                    dst.write(src.read())

                self.logger.info(f"Archived {sink_name} to {archive_name}")

            # Truncate the original file (atomic operation)
            sink_file.write_text("", encoding="utf-8")

            self.logger.info(f"Rotated {sink_name} - file truncated")

        except Exception as e:
            self.logger.error(f"Failed to rotate {sink_name}: {e}")

    def _export_leads(self, lead_records: List[Dict[str, Any]]) -> None:
        """Export leads to JSONL and CSV formats."""
        try:
            exports_config = self.icp_config.get("exports", {})
            if not exports_config.get("enabled", False):
                return

            # Create export directory with UTC timestamp
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
            export_dir = (Path(self.data_dir) / "exports" /
                          self.icp_config["name"] / timestamp)
            export_dir.mkdir(parents=True, exist_ok=True)

            # Export to JSONL
            jsonl_file = export_dir / "leads.jsonl"
            with open(jsonl_file, "w", encoding="utf-8") as f:
                for record in lead_records:
                    f.write(json.dumps(record) + "\n")

            # Export to CSV
            csv_file = export_dir / "leads.csv"
            self._export_to_csv(lead_records, csv_file)

            self.logger.info(
                f"Exported {len(lead_records)} leads to {export_dir}")

        except Exception as e:
            self.logger.error(f"Failed to export leads: {e}")
            self.total_errors.append(f"Export failed: {e}")

    def _export_to_csv(self, lead_records: List[Dict[str, Any]],
                       csv_file: Path) -> None:
        """Export leads to CSV format with proper flattening rules."""
        if not lead_records:
            return

        # Get all possible fields for CSV headers
        all_fields = set()
        for record in lead_records:
            self._collect_fields(record, "", all_fields)

        # Sort fields lexicographically for stability
        sorted_fields = sorted(all_fields)

        # Write CSV header
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            f.write(",".join(f'"{field}"' for field in sorted_fields) + "\n")

            # Write data rows
            for record in lead_records:
                row_data = []
                for field in sorted_fields:
                    value = self._get_field_value(record, field)
                    row_data.append(f'"{value}"')
                f.write(",".join(row_data) + "\n")

    def _collect_fields(self, obj: Any, prefix: str, fields: set) -> None:
        """Recursively collect all field paths for CSV flattening."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                field_path = f"{prefix}.{key}" if prefix else key

                # Special handling for contacts array - include best contact fields
                if key == "contacts" and isinstance(value, list) and value:
                    self._add_best_contact_fields(fields)
                elif isinstance(value, (dict, list)):
                    self._collect_fields(value, field_path, fields)
                else:
                    fields.add(field_path)
        elif isinstance(obj, list):
            # For arrays, join with "|" separator
            if (obj and len(obj) > 0 and isinstance(obj[0],
                                                    (str, int, float, bool))
                    or (obj and len(obj) > 0 and obj[0] is None)):
                field_path = prefix
                fields.add(field_path)

    def _add_best_contact_fields(self, fields: set) -> None:
        """Add best contact fields to CSV export."""
        contact_fields = [
            "best_contact.full_name",
            "best_contact.job_title",  # Changed from role to job_title
            "best_contact.email",
            "best_contact.phone",
            "best_contact.decision_maker",
            "best_contact.linkedin_url",  # Changed from linkedin to linkedin_url
            "best_contact.department",
            "best_contact.seniority",
            "best_contact.location",
        ]
        fields.update(contact_fields)

    def _get_best_contact_field(self, record: Dict[str, Any],
                                field_name: str) -> str:
        """Get field value from the best contact."""
        try:
            # Get best contact ID
            best_contact_id = record.get("best_contact_id")
            contacts = record.get("contacts", [])

            if not contacts:
                return ""

            # If no best_contact_id is set, use the first contact
            if not best_contact_id and contacts:
                best_contact = contacts[0]
            else:
                # Find the best contact in contacts array by ID
                best_contact = None
                for contact in contacts:
                    if contact.get("contact_id") == best_contact_id:
                        best_contact = contact
                        break

                if not best_contact:
                    return ""

            # Get the requested field value
            # Handle field name mapping for backward compatibility
            if field_name == "role":
                value = best_contact.get("role") or best_contact.get(
                    "job_title")
            elif field_name == "linkedin":
                value = best_contact.get("linkedin") or best_contact.get(
                    "linkedin_url")
            elif field_name == "full_name":
                # Construct full name from first_name and last_name if not present
                value = best_contact.get("full_name")
                if not value:
                    first_name = best_contact.get("first_name", "")
                    last_name = best_contact.get("last_name", "")
                    if first_name or last_name:
                        value = f"{first_name} {last_name}".strip()
            else:
                value = best_contact.get(field_name)

            if value is None:
                return ""
            elif isinstance(value, bool):
                return str(value).lower()
            elif isinstance(value, list):
                return "|".join(str(v) if v is not None else "" for v in value)
            else:
                return str(value)
        except Exception:
            return ""

    def _get_field_value(self, record: Dict[str, Any], field_path: str) -> str:
        """Get field value from record using dot notation."""
        try:
            keys = field_path.split(".")

            # Special handling for best_contact fields
            if keys[0] == "best_contact" and len(keys) > 1:
                return self._get_best_contact_field(record, keys[1])

            value = record
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return ""

            if value is None:
                return ""
            elif isinstance(value, bool):
                return str(value).lower()
            elif isinstance(value, list):
                # Join arrays with "|" separator
                if (value and len(value) > 0
                        and isinstance(value[0], (str, int, float, bool))
                        or (value and len(value) > 0 and value[0] is None)):
                    return "|".join(
                        str(v) if v is not None else "" for v in value)
                else:
                    return ""  # Skip arrays of objects
            else:
                return str(value)
        except Exception:
            return ""

    def _calculate_final_metrics(
        self,
        domain_result: Dict[str, Any],
        crawl_result: Optional[Dict[str, Any]],
        enrich_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate final pipeline metrics."""
        duration_ms = (int(
            (time.time() - self.start_time) * 1000) if self.start_time else 0)

        # Traditional mode: domain_result is actually domain_result
        # Aggregate cost metrics (Domain + Crawl only)
        total_domain_cost = (domain_result.get("metrics",
                                               {}).get("cost_usd",
                                                       {}).get("domain", 0.0))
        total_crawl_cost = (crawl_result.get("metrics", {}).get(
            "cost_usd", {}).get("crawl", 0.0) if crawl_result else 0.0)

        # Count leads at each stage
        domain_count = len(
            domain_result.get("data", {}).get("lead_records", []))
        crawl_count = (len(
            crawl_result.get("data", {}).get("lead_records", []))
                       if crawl_result else 0)

        # Calculate pass rates
        domain_pass_rate = domain_result.get("metrics", {}).get("pass_rate")
        crawl_pass_rate = (crawl_result.get("metrics", {}).get("pass_rate")
                           if crawl_result else None)
        return {
            "duration_ms": duration_ms,
            "lead_counts": {
                "domain": domain_count,
                "crawl": crawl_count,
            },
            "pass_rates": {
                "domain": domain_pass_rate,
                "crawl": crawl_pass_rate,
            },
            "cost_usd": {
                "domain": round4(total_domain_cost),
                "crawl": round4(total_crawl_cost),
                "total": round4(total_domain_cost + total_crawl_cost),
            },
            "health": {
                "unknown_error_count": self.total_unknown_errors
            },
        }

    def _build_error_metrics(self) -> Dict[str, Any]:
        """Build metrics for error cases."""
        duration_ms = (int(
            (time.time() - self.start_time) * 1000) if self.start_time else 0)
        return {
            "duration_ms": duration_ms,
            "lead_counts": {
                "domain": 0,
                "crawl": 0
            },
            "pass_rates": {
                "domain": None,
                "crawl": None
            },
            "cost_usd": {
                "domain": 0.0,
                "crawl": 0.0,
                "total": 0.0
            },
            "health": {
                "unknown_error_count": self.total_unknown_errors
            },
        }

    def _log_pipeline_summary(self, metrics: Dict[str, Any]) -> None:
        """Log pipeline completion summary."""
        duration_seconds = metrics["duration_ms"] / 1000

        # Traditional mode summary (Domain → Crawl only)
        domains_per_hour = ((metrics["lead_counts"]["domain"] /
                             duration_seconds *
                             3600) if duration_seconds > 0 else 0)
        cost_per_crawled = ((metrics["cost_usd"]["total"] /
                             metrics["lead_counts"]["crawl"])
                            if metrics["lead_counts"]["crawl"] > 0 else 0)

        self.logger.info(
            f"Traditional pipeline completed in {duration_seconds:.2f}s")
        self.logger.info(
            f"Processed {metrics['lead_counts']['domain']} domains, {metrics['lead_counts']['crawl']} enriched"
        )
        self.logger.info(f"Performance: {domains_per_hour:.1f} domains/hour")
        self.logger.info(
            f"Cost: ${metrics['cost_usd']['total']:.4f} (${cost_per_crawled:.4f}/qualified lead)"
        )
        self.logger.info(
            f"Errors: {len(self.total_errors)} total, {self.total_unknown_errors} unknown"
        )


# ============================================================================
# CLI Interface
# ============================================================================


async def main():
    """Main CLI entry point."""
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(description="Lead Sorcerer Orchestrator")
    parser.add_argument("--config",
                        required=True,
                        help="Path to icp_config.json")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for processing",
    )
    parser.add_argument("--data-dir", help="Override data directory")

    args = parser.parse_args()

    # Set data directory if provided
    if args.data_dir:
        os.environ["LEADPOET_DATA_DIR"] = args.data_dir

    try:
        # Initialize and run orchestrator
        orchestrator = LeadSorcererOrchestrator(args.config, args.batch_size)
        result = await orchestrator.run_pipeline()

        if result["success"]:
            print(json.dumps(result, indent=2))
            return 0
        else:
            print(json.dumps(result, indent=2))
            return 1
    except Exception as e:
        print(json.dumps({
            "error": str(e),
            "success": False
        }, indent=2),
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
