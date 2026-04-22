from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union

# Incident API Schemas
class Incident(BaseModel):
    id: str
    title: str
    severity: str
    status: Union[int, str] # Drift makes this a string sometimes
    service: str
    created_at: float

# Metrics API Schemas
class DataPoint(BaseModel):
    timestamp: float
    value: Optional[float] = None
    metric_value: Optional[float] = None # Drift renames value to metric_value

class MetricSeries(BaseModel):
    metric_name: str
    service: str
    datapoints: List[DataPoint]

# Deploy API Schemas
class Deployment(BaseModel):
    deployment_id: str
    service: str
    version: str
    status: str
    timestamp: float

# Log API Schemas
class LogEntry(BaseModel):
    timestamp: float
    level: str
    service: str
    message: str
    trace_id: Optional[str] = None

class LogQueryResponse(BaseModel):
    logs: List[LogEntry]
    next_offset: Optional[int] = None
    next_cursor: Optional[str] = None # Drift changes pagination to cursor

# Notify API Schemas
class NotificationRequest(BaseModel):
    channel: str
    message: str
    priority: str

class NotificationResponse(BaseModel):
    success: bool
    message_id: Optional[str] = None
    rate_limit_remaining: int
