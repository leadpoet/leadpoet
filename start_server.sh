#!/bin/bash
export DEMO_MODE=true
exec uvicorn cloudrun_api.main:app --host 0.0.0.0 --port 5000
