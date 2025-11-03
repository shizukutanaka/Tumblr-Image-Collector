"""License and entitlement management for Tumblr Image Collector."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Any, Set

logger = logging.getLogger(__name__)


class LicenseStatus(str, Enum):
    """Represents current entitlement status."""

    ACTIVE = "active"
    EXPIRED = "expired"
    TRIAL = "trial"
    NONE = "none"


@dataclass
class LicenseInfo:
    """Current license information."""

    status: LicenseStatus
    plan_key: Optional[str] = None
    current_period_end: Optional[str] = None
    customer_email: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    stripe_customer_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LicenseManager:
    """Persist and validate licensing information."""

    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self._license: LicenseInfo = LicenseInfo(status=LicenseStatus.NONE)
        self._processed_events_path = self.storage_path.parent / "processed_events.json"
        self._processed_events: Set[str] = set()
        self._load()
        self._load_processed_events()

    def _load(self) -> None:
        if not self.storage_path.exists():
            logger.debug("License storage file not found at %s", self.storage_path)
            return

        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            status = LicenseStatus(data.get("status", LicenseStatus.NONE))
            self._license = LicenseInfo(
                status=status,
                plan_key=data.get("plan_key"),
                current_period_end=data.get("current_period_end"),
                customer_email=data.get("customer_email"),
                stripe_subscription_id=data.get("stripe_subscription_id"),
                stripe_customer_id=data.get("stripe_customer_id"),
                metadata=data.get("metadata") or {},
            )
            logger.debug("Loaded license info: %s", self._license)
        except Exception as exc:
            logger.error("Failed to load license info: %s", exc)
            self._license = LicenseInfo(status=LicenseStatus.NONE)

    def _save(self) -> None:
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            payload = asdict(self._license)
            if payload.get("metadata") is None:
                payload["metadata"] = {}
            self.storage_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.debug("License info saved to %s", self.storage_path)
        except Exception as exc:
            logger.error("Failed to save license info: %s", exc)

    def _load_processed_events(self) -> None:
        if not self._processed_events_path.exists():
            return
        try:
            data = json.loads(self._processed_events_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._processed_events = set(map(str, data))
        except Exception as exc:
            logger.error("Failed to load processed Stripe events: %s", exc)
            self._processed_events = set()

    def _save_processed_events(self) -> None:
        try:
            self._processed_events_path.parent.mkdir(parents=True, exist_ok=True)
            self._processed_events_path.write_text(
                json.dumps(sorted(self._processed_events), ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except Exception as exc:
            logger.error("Failed to persist processed Stripe events: %s", exc)

    def set_license(self, info: LicenseInfo) -> None:
        self._license = info
        self._save()

    def update_status(self, status: LicenseStatus, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self._license, key):
                setattr(self._license, key, value)
        self._license.status = status
        self._save()

    def get_license(self) -> LicenseInfo:
        return self._license

    def is_active(self) -> bool:
        return self._license.status == LicenseStatus.ACTIVE

    def requires_subscription(self) -> bool:
        """Check if subscription features may be used."""

        return self.is_active() and self._license.plan_key is not None

    def clear(self) -> None:
        self._license = LicenseInfo(status=LicenseStatus.NONE)
        self._save()

    # --- Stripe subscription lifecycle helpers ---

    def has_processed_event(self, event_id: str) -> bool:
        return event_id in self._processed_events

    def record_processed_event(self, event_id: str) -> None:
        if not event_id:
            return
        self._processed_events.add(str(event_id))
        self._save_processed_events()

    def apply_subscription(
        self,
        plan_key: str,
        subscription_id: str,
        customer_email: Optional[str] = None,
        current_period_end: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        stripe_customer_id: Optional[str] = None,
    ) -> None:
        metadata = metadata or {}
        new_license = LicenseInfo(
            status=LicenseStatus.ACTIVE,
            plan_key=plan_key,
            current_period_end=current_period_end,
            customer_email=customer_email,
            stripe_subscription_id=subscription_id,
            stripe_customer_id=stripe_customer_id,
            metadata=metadata,
        )
        self.set_license(new_license)

    def update_subscription_period(
        self,
        subscription_id: str,
        current_period_end: Optional[str],
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._license.stripe_subscription_id != subscription_id:
            logger.debug(
                "Ignoring subscription period update for non-matching subscription_id: %s", subscription_id
            )
            return

        if current_period_end:
            self._license.current_period_end = current_period_end
        if metadata_updates:
            base_metadata = self._license.metadata or {}
            base_metadata.update(metadata_updates)
            self._license.metadata = base_metadata
        self._license.status = LicenseStatus.ACTIVE
        self._save()

    def expire_subscription(
        self,
        subscription_id: Optional[str],
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> None:
        if subscription_id and self._license.stripe_subscription_id != subscription_id:
            logger.debug(
                "Ignoring subscription expiration for non-matching subscription_id: %s", subscription_id
            )
            return
        if metadata_updates:
            base_metadata = self._license.metadata or {}
            base_metadata.update(metadata_updates)
            self._license.metadata = base_metadata
        self._license.status = LicenseStatus.EXPIRED
        self._license.current_period_end = None
        self._license.plan_key = None
        self._license.stripe_subscription_id = None
        self._license.stripe_customer_id = None
        self._save()
