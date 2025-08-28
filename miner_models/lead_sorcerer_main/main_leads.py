"""
Integration wrapper for Lead Sorcerer model to be compatible with the existing miner system.

This module provides a get_leads() function that runs the Lead Sorcerer orchestrator
and converts the output to the format expected by the existing miner code.
"""

import asyncio
import json
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Check for required dependencies first
def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import phonenumbers
        import httpx
        import openai
        return True, None
    except ImportError as e:
        return False, str(e)

# Check dependencies before importing Lead Sorcerer components
deps_ok, error_msg = check_dependencies()
if not deps_ok:
    print(f"❌ Could not import Lead Sorcerer orchestrator: {error_msg}")
    print("   Please ensure the Lead Sorcerer model is properly installed")
    print("   Run: pip install -r miner_models/lead_sorcerer_main/requirements.txt")
    
    # Provide fallback function that returns empty results
    async def get_leads(num_leads: int, industry: str = None, region: str = None) -> List[Dict[str, Any]]:
        """Fallback function when dependencies are missing."""
        print("⚠️ Lead Sorcerer dependencies not available, returning empty results")
        return []
else:
    # Get the absolute path to the lead_sorcerer_main directory
    lead_sorcerer_dir = Path(__file__).parent.absolute()
    src_path = lead_sorcerer_dir / "src"
    config_path = lead_sorcerer_dir / "config"
    
    # Add the src directory to the path so we can import the orchestrator
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    try:
        # Try to import with absolute path first
        import sys
        old_path = sys.path.copy()
        
        # Add both the lead_sorcerer_main directory and its src subdirectory
        sys.path.insert(0, str(lead_sorcerer_dir))
        sys.path.insert(0, str(src_path))
        
        from orchestrator import LeadSorcererOrchestrator
        LEAD_SORCERER_AVAILABLE = True
        
        # Restore original path but keep our additions
        for path in [str(lead_sorcerer_dir), str(src_path)]:
            if path not in old_path and path in sys.path:
                continue  # Keep our additions
                
    except ImportError as e:
        print(f"❌ Could not import Lead Sorcerer orchestrator: {e}")
        print(f"   Tried to import from: {src_path}")
        print(f"   Directory exists: {src_path.exists()}")
        if src_path.exists():
            print(f"   Contents: {list(src_path.iterdir())}")
        LEAD_SORCERER_AVAILABLE = False

    # Suppress verbose logging from the lead sorcerer
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    # ───────────────────────────────────────────────────────────────────
    #  Load canonical ICP template (must exist)
    # ───────────────────────────────────────────────────────────────────
    try:
        ICP_TEMPLATE_PATH = lead_sorcerer_dir / "icp_config.json"
        with open(ICP_TEMPLATE_PATH, "r", encoding="utf-8") as _f:
            BASE_ICP_CONFIG: Dict[str, Any] = json.load(_f)
    except Exception as e:
        raise RuntimeError(
            f"Lead Sorcerer wrapper: required icp_config.json not found or unreadable "
            f"at {ICP_TEMPLATE_PATH}. Error: {e}"
        ) from e


    def create_industry_specific_config(industry: str | None = None) -> Dict[str, Any]:
        """
        Clone the canonical icp_config.json and (optionally) tweak `icp_text`
        and `queries` if the caller requested a specific industry.

        NOTE: There is deliberately NO generic default – if the template file
        is absent we abort early.
        """
        config = json.loads(json.dumps(BASE_ICP_CONFIG))   # deep-copy

        if not industry:
            return config

        ind = industry.lower()

        # minimal heuristic tweak (keeps the rest of the template intact)
        if any(k in ind for k in ("tech", "software", "ai")):
            config["icp_text"] = "Technology companies needing contacts."
            config["queries"]  = ["technology company contact information"]
        elif any(k in ind for k in ("finance", "fintech", "bank")):
            config["icp_text"] = "Finance / FinTech organisations needing contacts."
            config["queries"]  = ["fintech company contact information"]
        elif any(k in ind for k in ("health", "med", "clinic")):
            config["icp_text"] = "Healthcare & wellness businesses needing contacts."
            config["queries"]  = ["healthcare company contact information"]
        # add more branches as desired …

        return config

    def setup_temp_environment(temp_dir: str):
        """Set up the temporary environment with required config files."""
        temp_path = Path(temp_dir)
        
        # Create config directory in temp
        temp_config_dir = temp_path / "config"
        temp_config_dir.mkdir(exist_ok=True)
        
        # Copy required config files
        source_config_dir = config_path
        
        # Copy costs.yaml (required)
        costs_file = source_config_dir / "costs.yaml"
        if costs_file.exists():
            shutil.copy2(costs_file, temp_config_dir / "costs.yaml")
        
        # Copy any other config files that might be needed
        for config_file in ["apollo_defaults.yaml"]:
            source_file = source_config_dir / config_file
            if source_file.exists():
                shutil.copy2(source_file, temp_config_dir / config_file)
        
        # Copy prompts directory if it exists
        source_prompts = source_config_dir / "prompts"
        if source_prompts.exists():
            temp_prompts = temp_config_dir / "prompts"
            shutil.copytree(source_prompts, temp_prompts, dirs_exist_ok=True)

        # NEW: copy the JSON-schema directory so validation works
        source_schemas = lead_sorcerer_dir / "schemas"
        if source_schemas.exists():
            temp_schemas = temp_path / "schemas"
            shutil.copytree(source_schemas, temp_schemas, dirs_exist_ok=True)

    def convert_lead_record_to_legacy_format(lead_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a Lead Sorcerer lead record to the format expected by the existing miner code.
        
        Args:
            lead_record: Lead record from Lead Sorcerer in unified schema format
            
        Returns:
            Lead in the format expected by the existing miner system
        """
        company = lead_record.get("company", {})
        contacts = lead_record.get("contacts", [])
        
        # Get the best contact (prefer one with a name, then any contact with an email)
        best_contact = None
        if contacts:
            # Look for a contact with both first and last name
            named_contacts = [c for c in contacts if c.get("first_name") and c.get("last_name")]
            if named_contacts:
                best_contact = named_contacts[0]
            else:
                # Otherwise use the first contact that has an email
                email_contacts = [c for c in contacts if c.get("email")]
                if email_contacts:
                    best_contact = email_contacts[0]
                elif contacts:
                    # Finally, just use the first contact
                    best_contact = contacts[0]
        
        # Extract contact information
        if best_contact:
            first_name = best_contact.get("first_name") or ""
            last_name = best_contact.get("last_name") or ""
            email = best_contact.get("email") or ""
            job_title = best_contact.get("job_title") or ""
            
            # Create full name from first and last name
            if first_name and last_name:
                full_name = f"{first_name} {last_name}"
            elif first_name:
                full_name = first_name
            elif last_name:
                full_name = last_name
            else:
                full_name = ""
        else:
            first_name = ""
            last_name = ""
            full_name = ""
            email = ""
            job_title = ""
        
        # Helper function to safely get string values
        def safe_str(value, default=""):
            """Safely convert value to string, handling None values."""
            if value is None:
                return default
            return str(value)
        
        # Build the enhanced format with all requested fields
        legacy_lead = {
            "Business": safe_str(company.get("name")),
            "description": safe_str(company.get("description")),
            "Owner Full name": full_name,
            "First": first_name,
            "Last": last_name,
            "Owner(s) Email": email,
            "phone_numbers": company.get("phone_numbers", []),
            "Website": f"https://{safe_str(lead_record.get('domain'))}" if lead_record.get('domain') else "",
            "Industry": safe_str(company.get("industry")),
            "sub_industry": safe_str(company.get("sector")),
            "role": job_title,
            "Region": safe_str(company.get("hq_location")),
            "Founded Year": safe_str(company.get("founded_year")),
            "Ownership Type": safe_str(company.get("ownership_type")),
            "Company Type": safe_str(company.get("company_type")),
            "Number of Locations": safe_str(company.get("number_of_locations")),
            "ids": company.get("ids", {}),
            "socials": company.get("socials", {}),
        }
        
        return legacy_lead

    async def run_lead_sorcerer_pipeline(num_leads: int, industry: str = None, region: str = None) -> List[Dict[str, Any]]:
        """
        Run the Lead Sorcerer pipeline and extract leads.
        
        Args:
            num_leads: Number of leads to generate
            industry: Target industry (optional)
            region: Target region (optional)
            
        Returns:
            List of lead records from Lead Sorcerer
        """
        if not LEAD_SORCERER_AVAILABLE:
            return []
            
        # Create a temporary directory for this run
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up the temporary environment with config files
            setup_temp_environment(temp_dir)
            
            # Set the data directory
            os.environ["LEADPOET_DATA_DIR"] = temp_dir
            
            # Change to temp directory so relative paths work
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                
                # Create configuration
                config = create_industry_specific_config(industry)
                
                # Adjust caps based on number of requested leads
                config["caps"]["max_domains_per_run"] = min(max(num_leads * 2, 5), 20)
                config["caps"]["max_crawl_per_run"] = min(max(num_leads * 2, 5), 20)
                config["caps"]["max_enrich_per_run"] = min(max(num_leads, 5), 15)
                
                # Save config to temporary file
                config_file = Path(temp_dir) / "icp_config.json"
                with open(config_file, "w") as f:
                    json.dump(config, f, indent=2)
                
                try:
                    # Initialize and run orchestrator
                    orchestrator = LeadSorcererOrchestrator(str(config_file), batch_size=num_leads)
                    
                    async with orchestrator:  # Use async context manager for proper cleanup
                        result = await orchestrator.run_pipeline()
                        
                        if not result.get("success"):
                            print(f"⚠️ Lead Sorcerer pipeline failed: {result.get('errors', [])}")
                            return []
                        
                        # Extract leads from the result - look in exports directory
                        leads = []
                        
                        # Look for exported leads in the exports directory
                        exports_dir = Path(temp_dir) / "exports"
                        if exports_dir.exists():
                            # Find the most recent export directory
                            export_dirs = list(exports_dir.glob("*/*"))
                            if export_dirs:
                                latest_export = max(export_dirs, key=lambda x: x.stat().st_mtime)
                                leads_file = latest_export / "leads.jsonl"
                                
                                if leads_file.exists():
                                    with open(leads_file, "r") as f:
                                        for line in f:
                                            if line.strip():
                                                try:
                                                    lead_record = json.loads(line)
                                                    # Include leads that have contacts (successful enrichment)
                                                    if (lead_record.get("contacts") and 
                                                        len(lead_record.get("contacts", [])) > 0 and
                                                        len(leads) < num_leads):
                                                        leads.append(lead_record)
                                                except json.JSONDecodeError:
                                                    continue
                        
                        # Fallback: also check the traditional locations
                        if not leads:
                            domain_pass_file = Path(temp_dir) / "domain_pass.jsonl"
                            apollo_pass_file = Path(temp_dir) / "apollo_pass.jsonl"
                            
                            # Try to read from apollo results first, then domain results
                            for results_file in [apollo_pass_file, domain_pass_file]:
                                if results_file.exists():
                                    with open(results_file, "r") as f:
                                        for line in f:
                                            if line.strip():
                                                try:
                                                    lead_record = json.loads(line)
                                                    # Only include leads that passed ICP checks and have contacts
                                                    if (lead_record.get("icp", {}).get("pre_pass") and 
                                                        lead_record.get("contacts") and
                                                        len(leads) < num_leads):
                                                        leads.append(lead_record)
                                                except json.JSONDecodeError:
                                                    continue
                                    
                                    if leads:
                                        break  # Found leads, don't need to check other files
                        
                        return leads[:num_leads]  # Return only the requested number
                        
                except Exception as e:
                    print(f"❌ Error running Lead Sorcerer pipeline: {e}")
                    return []
                    
            finally:
                # Always restore the original working directory
                os.chdir(original_cwd)

    async def get_leads(num_leads: int, industry: str = None, region: str = None) -> List[Dict[str, Any]]:
        """
        Generate leads using the Lead Sorcerer model.
        
        This function is compatible with the existing miner system and can be used as a drop-in
        replacement for the get_leads function from miner_models.get_leads.
        
        Args:
            num_leads: Number of leads to generate
            industry: Target industry (optional)
            region: Target region (optional)
            
        Returns:
            List of leads in the format expected by the existing miner system
        """
        # Check if required environment variables are set
        required_env_vars = ["GSE_API_KEY", "GSE_CX", "OPENROUTER_KEY", "FIRECRAWL_KEY"]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"⚠️ Lead Sorcerer missing required environment variables: {missing_vars}")
            print("   Please set these in your .env file or environment")
            return []
        
        if not LEAD_SORCERER_AVAILABLE:
            print("⚠️ Lead Sorcerer not available, returning empty results")
            return []
        
        try:
            # Run the Lead Sorcerer pipeline
            lead_records = await run_lead_sorcerer_pipeline(num_leads, industry, region)
            
            if not lead_records:
                print("⚠️ Lead Sorcerer produced no leads")
                return []
            
            # Convert to legacy format
            legacy_leads = []
            for record in lead_records:
                try:
                    legacy_lead = convert_lead_record_to_legacy_format(record)
                    
                    # Only include leads with valid email and business name
                    if legacy_lead.get("Owner(s) Email") and legacy_lead.get("Business"):
                        legacy_leads.append(legacy_lead)
                        
                except Exception as e:
                    print(f"⚠️ Error converting lead record: {e}")
                    continue
            
            print(f"✅ Lead Sorcerer produced {len(legacy_leads)} valid leads")
            return legacy_leads
            
        except Exception as e:
            print(f"❌ Lead Sorcerer error: {e}")
            return []

# Fallback function if dependencies are not available
if not deps_ok:
    async def get_leads(num_leads: int, industry: str = None, region: str = None) -> List[Dict[str, Any]]:
        """Fallback function when dependencies are missing."""
        print("⚠️ Lead Sorcerer dependencies not available, returning empty results")
        return []

# For backward compatibility and testing
if __name__ == "__main__":
    # Test the function
    import time
    
    async def test_async():
        start_time = time.time()
        
        print("🧪 Testing Lead Sorcerer integration...")
        test_leads = await get_leads(2, "Technology")
        
        print(f"⏱️ Generated {len(test_leads)} leads in {time.time() - start_time:.2f}s")
        
        for i, lead in enumerate(test_leads, 1):
            print(f"\n{i}. {lead.get('Business', 'Unknown')}")
            print(f"   Contact: {lead.get('Owner Full name', 'Unknown')} ({lead.get('Owner(s) Email', 'No email')})")
            print(f"   Industry: {lead.get('Industry', 'Unknown')}")
            print(f"   Website: {lead.get('Website', 'No website')}")
    
    asyncio.run(test_async())
