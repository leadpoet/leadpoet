import json
import logging

def safe_json_load(path: str):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logging.debug(f"{path}: empty or malformed – returning []")
        return []
