# SPDX-License-Identifier: Apache-2.0
"""LMCache MP server module wiring the journal recorder and startup replayer."""

# Future
from __future__ import annotations

# Standard
from typing import Optional

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.multiprocess.cache_control.key_resolver import (
    resolve_grouped_object_keys,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.ext_server_module import (
    ExtServerModuleBuildContext,
    ExtServerModuleComponents,
)
from lmcache.v1.multiprocess.warm_prefetch import (
    COMPLETED,
    PENDING,
    WarmPrefetchJobs,
)
from nas_kv_warmup.config import WarmupConfig
from nas_kv_warmup.journal import (
    JournalWriter,
    WarmSequence,
    build_sequences,
    compact_journal,
    load_journal,
)
from nas_kv_warmup.recorder import JournalRecorder
from nas_kv_warmup.replayer import VLLMClient, WarmupReplayer

logger = init_logger(__name__)
# journal / recorder / replayer use plain ``logging`` so they import without
# LMCache; give their parent logger LMCache's handler and LMCACHE_LOG_LEVEL.
init_logger("nas_kv_warmup")


class LMCacheWarmupBackend:
    """:class:`WarmupBackend` on top of the MP server's shared context."""

    def __init__(self, ctx: MPCacheServerContext) -> None:
        self._ctx = ctx
        self._jobs = WarmPrefetchJobs()

    def is_registered(self, model: str, world_size: int) -> bool:
        registry = self._ctx.layout_desc_registry
        return bool(registry.find_group_layout_descs(model, world_size))

    def submit_prefetch(self, seq: WarmSequence) -> Optional[str]:
        ctx = self._ctx
        registry = ctx.layout_desc_registry
        group_layout_descs = registry.find_group_layout_descs(seq.model, seq.ws)
        if not group_layout_descs:
            return None
        try:
            attn_desc = registry.find_attn_desc(seq.model, seq.ws)
        except ValueError:
            return None

        token_ids = list(seq.token_ids)
        hashes = [h.hex() for h in ctx.token_hasher.compute_chunk_hashes(token_ids)]
        if tuple(hashes) != seq.chunk_hashes:
            # chunk_size or hash_algorithm differs from the run that wrote it.
            return None

        key_groups, chunks = resolve_grouped_object_keys(
            ctx.token_hasher,
            seq.model,
            seq.ws,
            token_ids,
            seq.salt,
            group_layout_descs,
            attn_desc,
        )
        if not chunks:
            return None
        return self._jobs.submit(ctx.storage_manager, key_groups)

    def poll_prefetch(self, job_id: str) -> Optional[tuple[int, int]]:
        status = self._jobs.poll(self._ctx.storage_manager, job_id)
        if status.state == PENDING:
            return None
        if status.state == COMPLETED:
            return status.found_keys, status.total_keys
        return 0, 0


class NasKVWarmupModule:
    """Records stored chunks to the NAS journal and warms vLLM at startup."""

    def __init__(self, ctx: MPCacheServerContext, cfg: WarmupConfig) -> None:
        self._ctx = ctx
        self._cfg = cfg
        self._recorder: Optional[JournalRecorder] = None
        self._replayer: Optional[WarmupReplayer] = None

        snapshot = load_journal(
            cfg.journal_dir, retention_s=cfg.journal_retention_days * 86400
        )
        if cfg.compact:
            try:
                compact_journal(cfg.journal_dir, snapshot, cfg.compact_min_idle_s)
            except OSError as exc:
                logger.warning("NAS KV journal compaction failed: %s", exc)
        records = snapshot.records()
        logger.info(
            "Loaded %d NAS KV journal records from %d files (%d malformed lines)",
            len(records),
            len(snapshot.files),
            snapshot.skipped_lines,
        )

        if cfg.record:
            if not ctx.event_bus.enabled:
                logger.warning(
                    "Observability event bus is disabled; the NAS KV journal "
                    "will not be recorded. Drop --disable-observability."
                )
            else:
                self._recorder = JournalRecorder(
                    JournalWriter(cfg.journal_dir),
                    EventType.MP_TOKENS,
                    (
                        EventType.L1_WRITE_FINISHED,
                        EventType.L1_WRITE_FINISHED_AND_READ_RESERVED,
                    ),
                    binding_cache_size=cfg.binding_cache_size,
                    flush_interval_s=cfg.flush_interval_s,
                    seen=(r.key for r in records),
                )
                self._recorder.attach(ctx.event_bus)
                logger.info("Recording NAS KV journal to %s", self._recorder.path)

        if cfg.replay:
            assert cfg.vllm_url is not None
            self._replayer = WarmupReplayer(
                build_sequences(records),
                LMCacheWarmupBackend(ctx),
                VLLMClient(cfg.vllm_url, cfg.request_timeout_s),
                cfg.replay_config(),
            )
            self._replayer.start()

    @property
    def context(self) -> MPCacheServerContext:
        return self._ctx

    def report_status(self) -> dict:
        return {
            "nas_kv_warmup": {
                "is_healthy": True,
                "journal_dir": self._cfg.journal_dir,
                "recorder": None if self._recorder is None else self._recorder.stats(),
                "replay": None if self._replayer is None else self._replayer.stats(),
            }
        }

    def close(self) -> None:
        if self._replayer is not None:
            self._replayer.stop()
        if self._recorder is not None:
            self._recorder.close()


def build_server_modules(
    ctx: ExtServerModuleBuildContext,
) -> ExtServerModuleComponents:
    cfg = WarmupConfig.from_dict(ctx.config)
    return ExtServerModuleComponents(
        modules=[NasKVWarmupModule(ctx.server_context, cfg)]
    )
