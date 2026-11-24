"""Deliver private collection transitions to the operator's macOS Notification Center."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import time


LOCATION = "/home/ec2-user/.local/share/leadpoet/collection/watch/status.json"
TITLE = "SN71 collection monitor"


def fetch():
    # Use the operator's existing short-lived SSO profile. The remote command is constant.
    aws = ["aws", "--profile", "leadpoet-lab", "--region", "us-east-1", "--no-cli-pager", "ssm"]
    instance = "i-07e945bb2653c2e8f"
    command = subprocess.check_output(
        [*aws, "send-command", "--instance-ids", instance, "--document-name", "AWS-RunShellScript",
         "--parameters", json.dumps({"commands": ["sudo -n -u ec2-user cat " + LOCATION]}),
         "--comment", "Read private SN71 collection watch", "--query", "Command.CommandId", "--output", "text"],
        text=True, stderr=subprocess.DEVNULL, timeout=30,
    ).strip()
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        response = subprocess.run([*aws, "get-command-invocation", "--instance-id", instance, "--command-id", command],
                                  capture_output=True, text=True, timeout=20)
        if response.returncode == 0:
            invocation = json.loads(response.stdout)
            if invocation["Status"] == "Success":
                break
            if invocation["Status"] not in ("Pending", "InProgress", "Delayed"):
                raise ValueError("gateway status unavailable")
        time.sleep(6)
    else:
        raise ValueError("gateway read timed out")
    value = json.loads(invocation["StandardOutputContent"])
    observed = datetime.fromisoformat(value["observed_at"])
    age = (datetime.now(timezone.utc) - observed).total_seconds()
    if not 0 <= age <= 25 * 60:
        raise ValueError("stale collection status")
    return value


def notify(message):
    # Pass the displayed text as data. No remote text is evaluated by AppleScript.
    subprocess.run(["/usr/bin/osascript", "-", TITLE, message],
                   input="on run argv\ndisplay notification (item 2 of argv) with title (item 1 of argv)\nend run",
                   text=True, check=True, capture_output=True, timeout=20)


def deliver(directory, fetch_status=fetch, notify_user=notify):
    os.umask(0o077)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / "latest.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    try:
        snapshot = fetch_status()
    except (ValueError, KeyError, OSError, subprocess.SubprocessError):
        if previous.get("delivery") == "offline":
            return
        notify_user("Gateway readback unavailable. Check the operator connection and collection watch timer.")
        result = dict(previous, delivery="offline")
    else:
        event = snapshot["event"]
        last = previous.get("snapshot", {}).get("event", {})
        if previous.get("delivery") != "online" or event != last:
            changes = list(event["changes"])
            detail = ", ".join(f"{name} = {event['changes'][name]}" for name in changes[:3])
            message = detail or "Private enrollment, worker and validator checks are running."
            if previous.get("delivery") == "offline":
                message = "Gateway readback restored. " + message
            if len(changes) > 3 or (last and event["number"] > last["number"] + 1):
                message += " Inspect latest.json and the gateway journal for all transitions."
            notify_user(message[:420])
        result = {"delivery": "online", "snapshot": snapshot}
    tmp = directory / ".latest.new"
    tmp.write_text(json.dumps(result, sort_keys=True))
    tmp.replace(path)


if __name__ == "__main__":
    deliver(Path.home() / "Library/Application Support/Leadpoet/collection-watch")
