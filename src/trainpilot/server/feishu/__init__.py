"""Feishu integration package for TrainPilot."""

from trainpilot.server.feishu.cards import (
    build_alert_card,
    build_milestone_card,
    build_resolved_card,
)
from trainpilot.server.feishu.client import FeishuCardClient, default_feishu_client

__all__ = [
    "build_alert_card",
    "build_milestone_card",
    "build_resolved_card",
    "FeishuCardClient",
    "default_feishu_client",
]
