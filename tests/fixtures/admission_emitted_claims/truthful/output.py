def company_row(record, icp):
    candidate = dict(record)
    if candidate.get("fit_status", "unknown") == "unknown":
        candidate["fit_status"] = "pass"
    if candidate["fit_status"] != "pass":
        return None
    return {
        "company_name": record["company_name"],
        "country": record.get("country"),
        "matched_icp_signal": 0,
        "source_url": record["source_url"],
        "date": record.get("date"),
        "description": record["quote"],
        "relevance": f'If {record["company_name"]} chooses to enter {icp["offering"]}, this could be relevant.',
    }
