```python
class GemmaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. 词嵌入层 (Embedding)
        # Gemma 的词表很大 (256,000)，这部分参数量占比很高
        self.embed_tokens = nn.Embedding(256000, config.width)
        
        # 2. 堆叠 18 层 Transformer Blocks
        self.layers = nn.ModuleList([
            GemmaBlock(config) for _ in range(config.depth)
        ])
        
        # 3. 最终归一化
        self.final_norm = nn.LayerNorm(config.width)
        
        # 4. 输出层 (LM Head)
        # 实际上 Gemma 使用 self.embed_tokens.weight 作为投影权重
        self.lm_head = nn.Linear(config.width, 256000, bias=False)

    def forward(self, input_ids):
        # input_ids shape: [batch_size, seq_len]
        x = self.embed_tokens(input_ids)
        
        # 逐层通过 Transformer
        for layer in self.layers:
            x = layer(x)
            
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

# --- 实例化并计算总参数量 ---

model_2b = GemmaModel(gemma_2b_cfg)
model_300m = GemmaModel(gemma_300m_cfg)

def total_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

print(f"Gemma 2B 完整网络参数量: {total_params(model_2b):.2f} M (约 2.5B)")
print(f"Gemma 300M 完整网络参数量: {total_params(model_300m):.2f} M (约 0.5B)")
```