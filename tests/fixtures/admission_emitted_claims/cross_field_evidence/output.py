def company_row(record, icp):
    if record.get("business_activity") != "develops therapeutic drug candidates in human clinical trials":
        return None
    return {
        "company_name": record["company_name"],
        "industry": "Clinical-Stage Therapeutics",
        "matched_icp_signal": 0,
        "source_url": record["source_url"],
        "date": record.get("date"),
        "description": record["quote"],
    }
