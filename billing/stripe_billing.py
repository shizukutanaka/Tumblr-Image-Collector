"""Stripe billing integration for Tumblr Image Collector."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Any, List

import stripe

logger = logging.getLogger(__name__)


@dataclass
class PricePlan:
    """Represents a price plan for checkout."""

    price_id: str
    name: str
    recurring: bool
    billing_period: Optional[str] = None
    features: Optional[List[str]] = None


class StripeBillingManager:
    """Manage Stripe billing and checkout sessions."""

    def __init__(self, api_key: str, success_url: str, cancel_url: str, plans: Dict[str, PricePlan]):
        self.api_key = api_key
        self.success_url = success_url
        self.cancel_url = cancel_url
        self.plans = plans
        stripe.api_key = api_key
        logger.debug("StripeBillingManager initialized with plans: %s", list(plans.keys()))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "StripeBillingManager":
        """Create a manager instance from configuration."""

        stripe_cfg = config.get("stripe", {})
        required_keys = ["secret_key", "success_url", "cancel_url", "plans"]
        missing = [key for key in required_keys if key not in stripe_cfg]
        if missing:
            raise ValueError(f"Missing Stripe configuration keys: {', '.join(missing)}")

        plans_cfg = stripe_cfg["plans"]
        if not isinstance(plans_cfg, dict) or not plans_cfg:
            raise ValueError("Stripe plans configuration must be a non-empty dictionary")

        plans = {}
        for plan_key, plan_info in plans_cfg.items():
            plan = PricePlan(
                price_id=plan_info["price_id"],
                name=plan_info.get("name", plan_key),
                recurring=plan_info.get("recurring", False),
                billing_period=plan_info.get("billing_period"),
                features=plan_info.get("features")
            )
            plans[plan_key] = plan

        return cls(
            api_key=stripe_cfg["secret_key"],
            success_url=stripe_cfg["success_url"],
            cancel_url=stripe_cfg["cancel_url"],
            plans=plans
        )

    def create_checkout_session(self, plan_key: str, customer_email: Optional[str] = None, metadata: Optional[Dict[str, str]] = None) -> stripe.checkout.Session:
        """Create a checkout session for the given plan."""

        if plan_key not in self.plans:
            raise ValueError(f"Unknown plan key: {plan_key}")

        plan = self.plans[plan_key]
        checkout_args: Dict[str, Any] = {
            "mode": "subscription" if plan.recurring else "payment",
            "line_items": [
                {
                    "price": plan.price_id,
                    "quantity": 1,
                }
            ],
            "success_url": self.success_url,
            "cancel_url": self.cancel_url,
            "metadata": metadata or {},
        }

        if plan.recurring:
            checkout_args["subscription_data"] = {
                "metadata": metadata or {}
            }

        if customer_email:
            checkout_args["customer_email"] = customer_email

        logger.debug("Creating checkout session for plan %s with args: %s", plan_key, checkout_args)
        session = stripe.checkout.Session.create(**checkout_args)
        return session

    def list_products(self) -> List[Dict[str, Any]]:
        """Return a simple description of plans for display in UI."""

        return [
            {
                "key": key,
                "name": plan.name,
                "recurring": plan.recurring,
                "billing_period": plan.billing_period,
                "features": plan.features or [],
            }
            for key, plan in self.plans.items()
        ]

    def to_json(self) -> str:
        """Serialize plan definitions to JSON."""

        payload = {
            key: {
                "price_id": plan.price_id,
                "name": plan.name,
                "recurring": plan.recurring,
                "billing_period": plan.billing_period,
                "features": plan.features or [],
            }
            for key, plan in self.plans.items()
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
