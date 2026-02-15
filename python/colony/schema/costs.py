from __future__ import annotations

from pydantic import BaseModel, Field


class ProcessingCosts(BaseModel):
    cpu: float = Field(default=0.0, description="CPU usage cost in USD")
    gpu: float = Field(default=0.0, description="GPU usage cost in USD")
    memory: float = Field(default=0.0, description="Memory usage cost in USD")
    network: float = Field(default=0.0, description="Network usage cost in USD")
    storage: float = Field(default=0.0, description="Storage usage cost in USD")
