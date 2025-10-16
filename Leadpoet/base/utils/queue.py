import json
import os
import threading

DATA_DIR = "data"
QUEUE_FILE = os.path.join(DATA_DIR, "prospect_queue.json")
_queue_lock = threading.Lock()

def initialize_queue():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(QUEUE_FILE):
        with open(QUEUE_FILE, "w") as f:
            json.dump([], f)

def enqueue_prospects(prospects,
                      miner_hotkey: str,
                      request_type: str = "sourced",
                      **meta):
    with _queue_lock:
        initialize_queue()
        with open(QUEUE_FILE, "r") as f:
            queue = json.load(f)
        queue.append({
            "prospects": prospects,
            "miner_hotkey": miner_hotkey,
            "request_type": request_type,
            **meta
        })
        with open(QUEUE_FILE, "w") as f:
            json.dump(queue, f, indent=2)

def dequeue_prospects():
    with _queue_lock:
        initialize_queue()
        with open(QUEUE_FILE, "r") as f:
            queue = json.load(f)
        if not queue:
            return None
        lead_request = queue.pop(0)
        with open(QUEUE_FILE, "w") as f:
            json.dump(queue, f, indent=2)
        return lead_request