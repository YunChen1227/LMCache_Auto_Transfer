# NAS KV Cache 启动预热插件（LMCache MP 模式）

`nas_kv_warmup` 是一个 LMCache MP server 的 server-module 插件，做两件事：

1. **记录**：服务期间，每当一个 KV chunk 写进 LMCache，就把它对应的 token 记到 NAS 上的 journal 里。NAS 上的 KV 文件只有哈希，没有 token，所以需要单独记录。
2. **预热**：下次启动时，从 journal 还原出完整的 token 前缀，先把对应的 KV 从 NAS 预取到 CPU（L1），再向 vLLM 自动发送 `max_tokens=1` 的请求。vLLM 命中 LMCache 后，KV 直接从 CPU 拷进显存，不做 prefill，之后这些 block 留在 vLLM 的前缀缓存里。

整个过程不需要你准备任何 prompt。插件不修改 LMCache 核心代码。

```text
服务期间:  vLLM --store--> LMCache --KV 文件--> NAS kv 目录
                             |
                             +--MP_TOKENS / L1_WRITE_FINISHED--> journal 记录器 --> NAS journal 目录

下次启动:  journal --> 回放器 --L2->L1 warm prefetch--> LMCache (从 NAS 读 KV)
                         |
                         +--POST /v1/completions max_tokens=1--> vLLM --命中 L1，载入显存
```

---

## 全流程：从安装到 `vllm serve`

下文用的示例路径和端口如下，按你的环境替换：

| 项 | 示例值 |
|---|---|
| NAS 挂载点 | `/mnt/nas` |
| KV 目录 / journal 目录 | `/mnt/nas/lmcache/kv` / `/mnt/nas/lmcache/journal` |
| 日志目录 | `/var/log/lmcache` |
| LMCache 请求端口 / HTTP 端口 | `5555` / `8080` |
| vLLM 端口 | `8000` |
| 模型路径 / 对外模型名 | `/models/Qwen3-14B` / `my-model` |

### 第 0 步：前置条件

- Linux 加 NVIDIA GPU，CUDA 驱动已就绪。
- Python 3.10–3.13。
- NAS 已挂载，并且**所有推理机器上的挂载路径相同**（KV 文件名里不含路径，但配置里写的是绝对路径）。

### 第 1 步：安装 vLLM 和 LMCache

MP 模式建议使用最新版的 vLLM 和 LMCache。任选一种方式：

```bash
# 方式 A：安装发布版
python3 -m venv ~/venv-lmcache && source ~/venv-lmcache/bin/activate
pip install -U uv
uv pip install vllm lmcache

# 方式 B：从源码安装 LMCache（需要 dev 分支上的新功能时）
git clone https://github.com/LMCache/LMCache.git && cd LMCache
uv pip install -r requirements/build.txt
uv pip install vllm
uv pip install -e . --no-build-isolation
```

检查是否装好：

```bash
python -c "import vllm, lmcache; print(vllm.__version__)"
lmcache server --help | grep -- --server-module
```

### 第 2 步：安装本插件

在**同一个** Python 环境里安装：

```bash
pip install -e <LMCache 仓库>/examples/nas_kv_warmup
python -c "import nas_kv_warmup; print('ok')"
```

### 第 3 步：准备 NAS 目录，并测一下读带宽

```bash
mkdir -p /mnt/nas/lmcache/kv /mnt/nas/lmcache/journal /var/log/lmcache
df -h /mnt/nas

# 测 NAS 顺序读带宽，用来估算预热耗时（测完删掉测试文件）
dd if=/dev/zero of=/mnt/nas/lmcache/ddtest bs=1M count=4096 oflag=direct
dd if=/mnt/nas/lmcache/ddtest of=/dev/null bs=1M iflag=direct
rm /mnt/nas/lmcache/ddtest
```

### 第 4 步：启动 LMCache server（终端 1）

```bash
export LMCACHE_LOG_LEVEL=INFO

lmcache server \
  --port 5555 --http-port 8080 \
  --l1-size-gb 60 --eviction-policy LRU \
  --l2-adapter '{"type":"fs","base_path":"/mnt/nas/lmcache/kv"}' \
  --server-module '{
    "module_path": "nas_kv_warmup",
    "config": {
      "journal_dir": "/mnt/nas/lmcache/journal",
      "vllm_url": "http://127.0.0.1:8000",
      "served_model_name": "my-model",
      "max_warm_tokens": 400000
    }
  }' 2>&1 | tee -a /var/log/lmcache/server.log
```

注意事项：

- `--l1-size-gb` 至少要能放下一批预热数据（`prefetch_batch_size` 个序列）。CPU 内存充裕的话，开大一些，未推进显存的 KV 命中时也更快。
- 每天的 `--chunk-size`、`--hash-algorithm` 要保持一致（不改就是默认值），否则前一天的 KV 无法命中。
- **不要**加 `--disable-observability`，否则插件无法记录 journal。
- `max_warm_tokens` 第一次可以先随便填，等第 5 步从 vLLM 日志拿到显存容量后再改。

启动后，server 日志里应该能看到：

```text
LMCache INFO: Loaded 0 NAS KV journal records from 0 files (0 malformed lines) (module.py:...)
LMCache INFO: Recording NAS KV journal to /mnt/nas/lmcache/journal/<hostname>-<pid>.jsonl (module.py:...)
LMCache INFO: NAS KV warmup: nothing to replay (replayer.py:...)        <- 第一次启动 journal 为空，这是正常的
```

### 第 5 步：启动 vLLM（终端 2）

```bash
vllm serve /models/Qwen3-14B \
  --served-model-name my-model \
  --port 8000 \
  --tensor-parallel-size 2 \
  --kv-transfer-config '{
    "kv_connector": "LMCacheMPConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "lmcache.mp.host": "tcp://localhost",
      "lmcache.mp.port": 5555
    }
  }' 2>&1 | tee -a /var/log/lmcache/vllm.log
```

注意事项：

- `--served-model-name` 必须和插件配置里的 `served_model_name` 一致，否则回放请求会返回 404。
- `kv_role` 必须是 `kv_both` 或 `kv_producer`，否则不会往 NAS 写 KV。只负责读缓存的部署可以用 `kv_consumer`，加载和预热照常可用，但不再写 NAS。
- 不要关闭 prefix caching（不要加 `--no-enable-prefix-caching`），否则推进显存的 KV 不会被保留。
- 不建议开 `lmcache.mp.lazy_offload`。如果要开，同时设置 `lmcache.mp.lazy_offload_max_deferral_seconds`，原因见“已知限制”。
- `--tensor-parallel-size` / `--pipeline-parallel-size` 决定了 KV 的分片方式，**白天和晚上必须一致**才能复用。

从 vLLM 日志里找出显存能放下多少 token，据此设置 `max_warm_tokens`（建议取 80%–90%），然后重启 LMCache server：

```bash
grep -E "GPU KV cache size|Maximum concurrency" /var/log/lmcache/vllm.log
# 例: GPU KV cache size: 450,000 tokens  ->  max_warm_tokens 设为 380000 左右
```

### 第 6 步：正常服务，确认 KV 和 journal 都写进了 NAS

先发一个长 prompt 试一下（至少要超过一个 chunk，默认 256 个 token）：

```bash
curl -s http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"my-model","prompt":"<一段足够长的文本>","max_tokens":16}' | jq .usage
```

然后从三个地方确认：

```bash
# 1. LMCache 日志：KV 已从显存存到 L1，随后会异步写进 NAS
grep "Stored .* tokens" /var/log/lmcache/server.log | tail

# 2. NAS 上出现了 KV 文件和 journal 记录
ls /mnt/nas/lmcache/kv | wc -l
wc -l /mnt/nas/lmcache/journal/*.jsonl

# 3. 插件的 recorder 统计
curl -s http://127.0.0.1:8080/status | jq '.nas_kv_warmup.recorder'
```

`recorder.recorded` 应该持续增长，`dropped`、`write_failures` 应该为 0。

### 第 7 步：切换部署或重启

1. 先把流量切走，等 30 秒到 1 分钟，让 L1→NAS 的异步写入完成。
2. 先停 vLLM（Ctrl-C），再停 LMCache server（Ctrl-C）。LMCache server 退出时，插件会把还没写盘的 journal 记录刷到 NAS。

### 第 8 步：下次启动，自动预热

按第 4 步、第 5 步的命令重新启动即可，配置不用改。LMCache server 启动后会读取 journal，等 vLLM 就绪后自动开始预热。查看进度和确认完成的方法见下一节。

---

## 如何确认 KV Cache 已全部推送完成

### 方法 1：看 LMCache server 日志（最直接）

实时跟踪预热进度：

```bash
tail -f /var/log/lmcache/server.log | grep --line-buffered "NAS KV warmup"
```

一次正常预热的日志依次是：

```text
LMCache INFO: Loaded 8123 NAS KV journal records from 3 files (0 malformed lines)
LMCache INFO: NAS KV warmup: 412 sequences loaded, waiting for vLLM to register its KV cache and pass /health
LMCache INFO: NAS KV warmup: replaying 300 of 412 sequences (1530000 tokens)
LMCache INFO: NAS KV warmup progress: 32/300 sequences processed, replayed=31 skipped_missing=1 tokens_warmed=158720
LMCache INFO: NAS KV warmup progress: 64/300 sequences processed, replayed=63 skipped_missing=1 tokens_warmed=322560
...
LMCache INFO: NAS KV warmup finished: {'state': 'done', 'sequences_total': 412, 'sequences_selected': 300, ..., 'replayed': 298, ..., 'elapsed_s': 95.4}
```

**出现 `NAS KV warmup finished` 且其中 `'state': 'done'`，就表示推送已经完成。** 只想看最终结果，可以执行：

```bash
grep "NAS KV warmup finished" /var/log/lmcache/server.log | tail -1
```

`replaying 300 of 412` 里，300 是被选中预热的序列数，412 是 journal 里能还原出的全部序列数。两者的差值，是超出 `max_warm_tokens` 预算或 TP×PP 不一致的序列，这些序列不会预热。

如果想看 NAS→CPU→显存每一段的细节，可以 grep 下面这些 LMCache 自带的日志：

```bash
# NAS -> CPU：每个序列预取到了多少 key，其中多少来自 L2（NAS）
grep "Prefetch request completed (L1+L2)" /var/log/lmcache/server.log | tail

# CPU -> 显存：vLLM 回放请求从 LMCache 取回了多少 token
grep "Retrieved .* tokens" /var/log/lmcache/server.log | tail
```

想看更多细节，可以把 `LMCACHE_LOG_LEVEL` 设为 `DEBUG` 后重启 LMCache server。

### 方法 2：查询 `/status` 接口

```bash
curl -s http://127.0.0.1:8080/status | jq '.nas_kv_warmup.replay'
```

返回示例：

```json
{
  "state": "done",
  "sequences_total": 412,
  "sequences_selected": 300,
  "skipped_other_world_size": 0,
  "skipped_missing": 2,
  "prefetch_timeouts": 0,
  "prefetch_errors": 0,
  "replayed": 298,
  "replay_failures": 0,
  "tokens_warmed": 1520000,
  "elapsed_s": 95.4
}
```

等待推送完成的脚本，可以放在部署流程里，预热完成后再接入流量：

```bash
while true; do
  state=$(curl -s http://127.0.0.1:8080/status | jq -r '.nas_kv_warmup.replay.state')
  echo "$(date +%T) warmup state: $state"
  case "$state" in
    done) echo "预热完成"; break ;;
    timeout|failed|stopped) echo "预热未完成: $state，请查看 server.log"; exit 1 ;;
  esac
  sleep 5
done
curl -s http://127.0.0.1:8080/status | jq '.nas_kv_warmup.replay'
```

**判断是否“全部推完”**：

- `state == "done"`：回放流程已结束。
- `replayed == sequences_selected`：所有选中的序列都推进了显存。
- 两者不相等时，差值等于 `skipped_missing + prefetch_timeouts + prefetch_errors + replay_failures`。各字段的含义见下表。

| 字段 | 含义 |
|---|---|
| `state` | `idle` → `waiting`（等 vLLM 就绪）→ `running`（推送中）→ `done`。异常时为 `timeout`、`failed` 或 `stopped` |
| `sequences_total` | journal 里能还原出的序列总数 |
| `sequences_selected` | 在 `max_warm_tokens` 预算内选中的序列数 |
| `replayed` / `tokens_warmed` | 成功推进显存的序列数 / token 数 |
| `skipped_missing` | NAS 上 KV 不全，或 `chunk_size`、哈希算法与写入时不一致而跳过的序列数 |
| `skipped_other_world_size` | TP×PP 和当前部署不同而跳过的序列数 |
| `prefetch_timeouts` | NAS→CPU 预取超过 `prefetch_timeout_s` 的序列数 |
| `prefetch_errors` / `replay_failures` | 预取提交失败 / 回放请求失败的序列数 |
| `elapsed_s` | 从启动预热到结束的耗时（包含等待 vLLM 就绪的时间） |

### 方法 3：在 vLLM 一侧验证真的命中了显存

预热完成后，发一个之前出现过的长 prompt：

```bash
# vLLM 的前缀缓存命中计数，发请求前后各看一次
curl -s http://127.0.0.1:8000/metrics | grep -E "^vllm:prefix_cache_(queries|hits)"

# vLLM 日志里定期打印的前缀缓存命中率
grep "Prefix cache hit rate" /var/log/lmcache/vllm.log | tail -3
```

命中显存时，`prefix_cache_hits` 会增加，而且 LMCache server 日志里**不会**因为这个请求新增 `Retrieved ... tokens`，因为 KV 已经在显存里，不需要再从 LMCache 取。

---

## 常见问题排查

| 现象 | 可能原因和处理 |
|---|---|
| `state` 一直是 `waiting`，最后变成 `timeout` | vLLM 没起来或 `vllm_url` 不对：`curl http://127.0.0.1:8000/health`。或者当前 TP×PP 和 journal 里所有记录都不一样（vLLM 注册的 world size 没有可回放的序列） |
| `skipped_other_world_size` 很大 | 今天的 TP×PP 和写入时不同，KV 分片对不上，只能重新积累缓存 |
| `skipped_missing` 很大 | NAS 上的 KV 文件被清理过；或 `--chunk-size`、`--hash-algorithm` 改过；或 `--l2-adapter` 的 `base_path` 和写入时不同 |
| `replay_failures` 大于 0 | 看 vLLM 日志。最常见的是 `served_model_name` 和 `--served-model-name` 不一致（404） |
| `prefetch_timeouts` 大于 0 | NAS 太慢或 L1 太小。调大 `prefetch_timeout_s`、`--l1-size-gb`，或调小 `prefetch_batch_size` |
| `recorder.recorded` 一直为 0 | vLLM 的 `kv_role` 不是 `kv_both`/`kv_producer`；或加了 `--disable-observability`（server 日志会有 warning）；或 prompt 都短于一个 chunk |
| `recorder.dropped` 大于 0 | event bus 或写盘队列满了，丢失的序列不会被预热，不影响正确性 |
| 日志里看不到 `NAS KV warmup` | 确认 `--server-module` 生效：`curl -s 127.0.0.1:8080/status \| jq 'has("nas_kv_warmup")'`；确认 `LMCACHE_LOG_LEVEL` 不高于 `INFO` |

---

## 配置项（`--server-module` 的 `config`）

| 键 | 默认值 | 说明 |
|---|---|---|
| `journal_dir` | 必填 | NAS 上的 journal 目录，所有实例共用 |
| `record` | `true` | 是否记录 journal |
| `replay` | `true` | 是否在启动时预热 |
| `vllm_url` | 回放时必填 | vLLM OpenAI 接口地址 |
| `served_model_name` | journal 中的模型名 | 请求里的 `model` 字段，需与 vLLM 的 `--served-model-name` 一致 |
| `max_warm_tokens` | `0`（不限） | 推进显存的 token 上限，共享前缀只计一次。建议设为 vLLM 日志里 `GPU KV cache size` 的 80%–90%，超出部分会被 LRU 淘汰，白白占用 NAS 带宽 |
| `prefetch_batch_size` | `32` | 每批先预取再回放的序列数，用来限制预热占用的 L1 |
| `replay_concurrency` | `8` | 同时向 vLLM 发送的回放请求数 |
| `world_size` | 自动 | 只回放该 TP×PP 下存储的序列；不填则回放 vLLM 已注册的 world size |
| `ready_timeout_s` | `1800` | 等待 vLLM 就绪的最长时间 |
| `prefetch_timeout_s` | `600` | 每批 NAS→CPU 预取的最长等待时间 |
| `request_timeout_s` | `600` | 单个回放请求的超时 |
| `journal_retention_days` | `0`（不限） | 早于该天数的 journal 记录不再回放，并在压缩时删除 |
| `compact` | `true` | 启动时合并空闲的 journal 文件 |
| `compact_min_idle_s` | `3600` | 只有超过这么久没被写入的 journal 文件才会被合并，避免动到其他在线实例正在写的文件 |
| `flush_interval_s` | `1.0` | journal 后台刷盘间隔 |
| `binding_cache_size` | `65536` | 内存中等待写完成事件的 token 绑定数上限 |

## 预热耗时

耗时主要由 NAS 读取带宽决定，CPU→显存那一段（PCIe）只要几秒。粗略估算：预热耗时 ≈ 实际预热的 KV 字节数 ÷ 第 3 步测出的 NAS 读带宽。这里的字节数受 `max_warm_tokens` 限制，不是 NAS 上的全部数据。实际耗时可以看 `/status` 里的 `elapsed_s`，或 `NAS KV warmup finished` 日志。

## 已知限制

- **journal 是尽力而为的**。event bus 队列满时会丢事件，也不能用 `--disable-observability` 关闭 observability bus（关闭时插件会打 warning，并且不记录）。丢失记录只意味着对应序列不会被预热，不影响正确性。
- **只回放完整 chunk**（`chunk_size` 的整数倍）。vLLM 仍会计算每个序列的最后一个 token，开销很小。
- **TP×PP、模型名、KV dtype、`chunk_size`、哈希算法必须与写入时一致**。回放前插件会用当前 server 的哈希重新计算 chunk 哈希，并做一次 NAS→CPU 预取；只有全部 key 都命中的序列才会发给 vLLM，所以不一致的序列不会被误触发 prefill。
- **lazy offload**：如果 vLLM 开启了 `lmcache.mp.lazy_offload`，停机前仍留在显存里、还没 offload 的 KV 不会写进 NAS，也就不会被预热。需要开启时，建议同时设置 `lmcache.mp.lazy_offload_max_deferral_seconds`，限制最多等待多久。
- **清理**：journal 会随时间增长，可用 `journal_retention_days` 加启动时压缩来控制。NAS 上的 KV 文件不归插件管，需要另行清理。KV 文件被删后，对应序列会在预取阶段被识别并跳过。
- 不支持多模态输入和 LoRA 请求的回放。

## 测试

测试不依赖 LMCache 或 GPU：

```bash
cd examples/nas_kv_warmup
pytest tests
```
