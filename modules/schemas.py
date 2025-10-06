"""Data schemas for Telegram parsing pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class ChannelConfig(BaseModel):
    """Configuration entry describing a Telegram channel."""

    username: str = Field(..., description="Публичный @username канала без @")
    title: Optional[str] = None
    enabled: bool = True

    @property
    def normalized_username(self) -> str:
        """Return username without leading @."""

        return self.username.lstrip("@")


class TelegramMessage(BaseModel):
    """Normalized representation of a Telegram message."""

    id: int
    channel_username: str
    channel_title: Optional[str] = None
    date_utc: datetime
    text: str
    raw_text: Optional[str] = None
    links: List[str] = Field(default_factory=list)
    mentions: List[str] = Field(default_factory=list)
    hashtags: List[str] = Field(default_factory=list)
    tickers: List[str] = Field(default_factory=list)
    is_forwarded: bool = False
    fwd_from: Optional[str] = None
    views: Optional[int] = None
    forwards: Optional[int] = None
    replies: Optional[int] = None
    msg_hash: str


__all__ = ["ChannelConfig", "TelegramMessage"]

