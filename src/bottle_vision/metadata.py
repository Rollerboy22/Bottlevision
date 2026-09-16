"""Typed metadata models shared by the pipeline and dataset layer."""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SampleMetadata(BaseModel):
    """Metadata required to keep a sample reproducible and auditable."""

    model_config = ConfigDict(extra="allow")

    sample_id: str
    image_path: str
    split: str | None = None
    pipeline_version: str = "0.1.0"
    model_version: str | None = None
    source: str = "unknown"
    status: str = "unverified"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    annotations: list[dict[str, Any]] = Field(default_factory=list)
