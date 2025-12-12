# core/trust_network.py
"""
A lightweight TrustNetwork helper that accepts both positional node ids and
keyword names such as target_node_id (fixes unexpected keyword arg errors).
This implementation is resilient and returns a float trust score in [0,1].
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
import math
import random


class TrustNetwork:
    def __init__(self, initial: Optional[Dict[int, float]] = None):
        # mapping node_id -> trust_score [0..1]
        self._trust: Dict[int, float] = dict(initial or {})

    def get(self, node_or_id: Union[int, Any], default: float = 0.5) -> float:
        """
        Accept either a node object with attribute node_id or an int id.
        """
        node_id = self._extract_node_id(node_or_id)
        return self._trust.get(node_id, default)

    def _extract_node_id(self, node_or_id: Union[int, Any]) -> Optional[int]:
        if node_or_id is None:
            return None
        if isinstance(node_or_id, int):
            return node_or_id
        # allow objects that carry node_id or id attribute
        node_id = getattr(node_or_id, "node_id", None)
        if node_id is None:
            node_id = getattr(node_or_id, "id", None)
        return node_id

    def update_trust(
        self,
        node: Optional[Union[int, Any]] = None,
        value: Optional[float] = None,
        *,
        node_id: Optional[int] = None,
        target_node_id: Optional[int] = None,
        delta: Optional[float] = None,
        signal_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Flexible update_trust signature to accept multiple calling conventions.

        Supported call styles:
            - update_trust(node_id, value)
            - update_trust(node=<node_obj>, value=0.7)
            - update_trust(target_node_id=42, value=0.6)
            - update_trust(node=..., delta=0.05)

        Returns the new trust score [0..1].
        """
        # determine target id
        target = None
        if target_node_id is not None:
            target = target_node_id
        elif node_id is not None:
            target = node_id
        elif node is not None:
            target = self._extract_node_id(node)

        if target is None:
            raise ValueError("No target node specified for trust update")

        current = self._trust.get(target, 0.5)

        # determine new value
        if value is not None:
            new = float(value)
        elif delta is not None:
            new = float(current + delta)
        else:
            # heuristics: use context and signal_type to nudge trust if provided
            adjustment = 0.0
            if context:
                # sample heuristic: successful exchange increases trust slightly
                outcome = context.get("outcome")
                if outcome == "success":
                    adjustment += 0.05
                elif outcome == "failure":
                    adjustment -= 0.08
                # energy and trust in context increase reliability
                energy_factor = context.get("source_energy", 0.0)
                adjustment += min(0.05, energy_factor * 0.01)
            # signal_type heuristics
            if signal_type == "warning":
                adjustment -= 0.02
            elif signal_type == "resource":
                adjustment += 0.03
            # add a small random smoothing term to avoid deterministic extremes
            adjustment += random.uniform(-0.01, 0.01)
            new = current + adjustment

        # clamp between 0 and 1
        new_clamped = max(0.0, min(1.0, new))
        self._trust[target] = new_clamped
        return new_clamped

    def get_trust_summary(self) -> Dict[str, Any]:
        if not self._trust:
            return {"count": 0, "avg_trust": None, "min": None, "max": None}
        vals = list(self._trust.values())
        return {
            "count": len(vals),
            "avg_trust": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
        }

    def reset(self) -> None:
        self._trust.clear()
