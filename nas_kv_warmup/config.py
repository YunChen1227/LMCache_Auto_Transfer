# SPDX-License-Identifier: Apache-2.0
"""Parsing of the ``--server-module`` ``config`` object."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Any, Optional

# First Party
from nas_kv_warmup.replayer import ReplayConfig


@dataclass(frozen=True)
class WarmupConfig:
    journal_dir: str
    record: bool = True
    replay: bool = True
    vllm_url: Optional[str] = None
    served_model_name: Optional[str] = None
    max_warm_tokens: int = 0
    replay_concurrency: int = 8
    ready_timeout_s: float = 1800.0
    prefetch_batch_size: int = 32
    prefetch_timeout_s: float = 600.0
    request_timeout_s: float = 600.0
    world_size: Optional[int] = None
    journal_retention_days: float = 0.0
    compact: bool = True
    compact_min_idle_s: float = 3600.0
    flush_interval_s: float = 1.0
    binding_cache_size: int = 65536

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WarmupConfig":
        """Validate and build the config.

        Raises:
            ValueError: A field is missing, unknown or has the wrong type.
        """
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown nas_kv_warmup config keys: {sorted(unknown)}")

        journal_dir = d.get("journal_dir")
        if not isinstance(journal_dir, str) or not journal_dir:
            raise ValueError("journal_dir must be a non-empty string")

        kwargs: dict[str, Any] = {"journal_dir": journal_dir}
        for name in ("record", "replay", "compact"):
            if name in d:
                kwargs[name] = _bool(d, name)
        for name in ("vllm_url", "served_model_name"):
            if d.get(name) is not None:
                kwargs[name] = _str(d, name)
        for name in (
            "max_warm_tokens",
            "replay_concurrency",
            "prefetch_batch_size",
            "binding_cache_size",
        ):
            if name in d:
                minimum = 0 if name == "max_warm_tokens" else 1
                kwargs[name] = _int(d, name, minimum=minimum)
        if d.get("world_size") is not None:
            kwargs["world_size"] = _int(d, "world_size", minimum=1)
        for name in (
            "ready_timeout_s",
            "prefetch_timeout_s",
            "request_timeout_s",
            "journal_retention_days",
            "compact_min_idle_s",
            "flush_interval_s",
        ):
            if name in d:
                kwargs[name] = _float(d, name)

        cfg = cls(**kwargs)
        if cfg.replay and not cfg.vllm_url:
            raise ValueError("vllm_url is required when replay is enabled")
        return cfg

    def replay_config(self) -> ReplayConfig:
        return ReplayConfig(
            served_model_name=self.served_model_name,
            max_warm_tokens=self.max_warm_tokens,
            replay_concurrency=self.replay_concurrency,
            ready_timeout_s=self.ready_timeout_s,
            prefetch_batch_size=self.prefetch_batch_size,
            prefetch_timeout_s=self.prefetch_timeout_s,
            world_size=self.world_size,
        )


def _bool(d: dict[str, Any], name: str) -> bool:
    value = d[name]
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _str(d: dict[str, Any], name: str) -> str:
    value = d[name]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _int(d: dict[str, Any], name: str, minimum: int) -> int:
    value = d[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _float(d: dict[str, Any], name: str) -> float:
    value = d[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return float(value)
