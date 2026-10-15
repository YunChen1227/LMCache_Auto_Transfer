# 测试对话数据（100 条，按输入长度分档）

文件：`conversations-100.jsonl`

| 档位 | 条数 | 每条 user 输入目标 |
|------|------|-------------------|
| 1k   | 20   | 1024 tokens       |
| 2k   | 20   | 2048 tokens       |
| 4k   | 20   | 4096 tokens       |
| 8k   | 20   | 8192 tokens       |
| 10k  | 20   | 10240 tokens      |

每行 JSON 字段：`id`、`tier`、`target_input_tokens`、`actual_input_tokens`、`messages`（仅一条 `user` 消息）。

## 重新生成

默认按「1 字符 ≈ 1 token」估算（中文场景与 Qwen 较接近）：

```bash
python testdata/generate_conversations.py
```

若要与线上模型完全一致，指定 tokenizer 后再生成：

```bash
export LMCACHE_TEST_TOKENIZER=/models/Qwen3-14B
python testdata/generate_conversations.py
```

## 批量请求 vLLM（chat）

```bash
VLLM=http://127.0.0.1:8000
MODEL=my-model
while IFS= read -r line; do
  curl -sS "$VLLM/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "$(python3 -c "
import json,sys
d=json.loads(sys.argv[1])
print(json.dumps({'model':sys.argv[2],'messages':d['messages'],'max_tokens':32,'temperature':0.7}))
" "$line" "$MODEL")" >/dev/null
  echo "id=$(echo "$line" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])') tier=$(echo "$line" | python3 -c 'import json,sys; print(json.load(sys.stdin)["tier"])')"
done < testdata/conversations-100.jsonl
```
