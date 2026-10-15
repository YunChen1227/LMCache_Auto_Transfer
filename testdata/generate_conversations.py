#!/usr/bin/env python3
"""Generate 100 dialogues with fixed input lengths (token targets).

Distribution (20 each): 1K, 2K, 4K, 8K, 10K total input tokens per line.

Each record is one ``user`` message (plus metadata). Token count = that message only.

Token counting:
  - Set env ``LMCACHE_TEST_TOKENIZER`` to a HuggingFace model path/name for exact counts.
  - Otherwise: one Unicode character ≈ one token (reasonable for Qwen + 中文).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

TIERS: list[tuple[str, int]] = [
    ("1k", 1024),
    ("2k", 2048),
    ("4k", 4096),
    ("8k", 8192),
    ("10k", 10240),
]
PER_TIER = 20

OPENERS = [
    "【用户】我想就下面的资料做一次完整梳理，请稍后根据这些内容回答我的问题。",
    "【用户】这是我们项目的背景材料，请先阅读并记住关键事实与数字。",
    "【用户】我在准备评审材料，请阅读以下说明，后续我会追问细节。",
    "【用户】请把下面内容当作对话上下文，阅读后等待我的进一步指令。",
]

SENTENCES = [
    "在企业级大模型部署中，前缀 KV 缓存复用可以显著降低重复文档问答的首 token 延迟。",
    "NAS 上的 KV 对象通常以 chunk 哈希命名，因此需要 journal 记录 token 与哈希的对应关系。",
    "预热流程应先在 L2 命中 KV，再通过极短 completion 将 block 推入 vLLM 前缀缓存。",
    "变更 tensor parallel 规模会导致 KV 分片变化，白天与夜间部署必须保持 world size 一致。",
    "监控应覆盖 journal 写入、warmup 成功率、跳过的缺失块数量以及 L1 占用率。",
    "长上下文压测时要控制并发，避免单批 prefetch 挤占 L1 影响在线流量。",
    "检索增强链路中，分块大小与 embedding 版本需要在配置中固定并版本化。",
    "异步 offload 未完成时重启可能丢最近一段 KV，切换前应留出 flush 窗口。",
]


def _heuristic_token_len(text: str) -> int:
    return len(text)


def _load_token_counter() -> tuple[Callable[[str], int], str]:
    model = os.environ.get("LMCACHE_TEST_TOKENIZER", "").strip()
    if model:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        def count(s: str) -> int:
            return len(tok.encode(s, add_special_tokens=False))

        return count, f"transformers:{model}"
    return _heuristic_token_len, "heuristic(char_len)"


def _fit_length(text: str, target: int, count: Callable[[str], int]) -> str:
    if count(text) <= target:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if count(text[:mid]) <= target:
            lo = mid + 1
        else:
            hi = mid
    end = max(0, lo - 1)
    while end > 0 and count(text[:end]) > target:
        end -= 1
    return text[:end]


def _build_user_content(dialog_id: int, target: int, count: Callable[[str], int]) -> str:
    opener = OPENERS[dialog_id % len(OPENERS)]
    parts = [f"对话编号 {dialog_id}。\n", opener, "\n\n"]
    i = 0
    while count("".join(parts)) < target:
        sent = SENTENCES[(dialog_id + i) % len(SENTENCES)]
        parts.append(f"（段落{i + 1}）{sent}\n")
        i += 1
    return _fit_length("".join(parts), target, count)


def make_dialogue(
    dialog_id: int,
    tier_label: str,
    target_tokens: int,
    count: Callable[[str], int],
) -> dict:
    content = _build_user_content(dialog_id, target_tokens, count)
    actual = count(content)
    return {
        "id": dialog_id,
        "target_input_tokens": target_tokens,
        "actual_input_tokens": actual,
        "tier": tier_label,
        "messages": [{"role": "user", "content": content}],
    }


def main() -> None:
    count, counter_name = _load_token_counter()
    out = Path(__file__).resolve().parent / "conversations-100.jsonl"
    records: list[dict] = []
    dialog_id = 1
    for tier_label, target in TIERS:
        for _ in range(PER_TIER):
            records.append(make_dialogue(dialog_id, tier_label, target, count))
            dialog_id += 1

    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {out} ({len(records)} dialogues)")
    print(f"Token counter: {counter_name}")
    for tier_label, target in TIERS:
        subset = [r for r in records if r["tier"] == tier_label]
        actuals = [r["actual_input_tokens"] for r in subset]
        print(
            f"  {tier_label}: target={target}, actual min/max={min(actuals)}/{max(actuals)}"
        )


if __name__ == "__main__":
    main()
