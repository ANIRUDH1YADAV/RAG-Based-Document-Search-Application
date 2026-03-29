"""
Route identifier model.
"""

from typing import Literal
from pydantic import BaseModel


class RouteIdentifier(BaseModel):
    """Model for routing queries to appropriate nodes."""

    route: Literal["index", "general", "search"]