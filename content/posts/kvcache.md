https://yuanchaofa.com/post/understanding-kv-cache-and-prompt-cache-basics#1-%E4%BB%80%E4%B9%88%E6%98%AF-kv-cache

```python
# 带 KV Cache 的生成
kv_cache = {}  # 每一层缓存 K, V
for step in range(max_new_tokens):
    if step == 0:
        # 第一步：处理所有 input tokens，填充 cache
        logits, kv_cache = model(input_tokens, kv_cache=None)
    else:
        # 后续步：只送入上一步生成的 1 个 token
        logits, kv_cache = model([last_token], kv_cache=kv_cache)
    next_token = sample(logits[-1])
    last_token = next_token

```