"""Gateway health response model."""

from typing import Any, Dict, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health check response."""

    service: str
    status: str
    build_id: str
    github_commit: str
    timestamp: str
    build_info: Optional[Dict[str, Any]] = None
