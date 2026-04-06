```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GemmaConfig:
    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width           # 隐藏层维度hidden_dim
        self.depth = depth           # 层数
        self.mlp_dim = mlp_dim       # MLP 中间层维度
        self.num_heads = num_heads   # Query 头数
        self.num_kv_heads = num_kv_heads # Key/Value 头数 (Gemma 为 1, 即 MQA)
        self.head_dim = head_dim     # 每个头的维度

class GemmaMLP(nn.Module):
    """Gemma 使用的是类似 SwiGLU 的门控线性单元"""
    def __init__(self, config):
        super().__init__()
        # 这里的维度变化是：width -> mlp_dim
        self.gate_proj = nn.Linear(config.width, config.mlp_dim, bias=False)
        self.up_proj = nn.Linear(config.width, config.mlp_dim, bias=False)
        self.down_proj = nn.Linear(config.mlp_dim, config.width, bias=False)

    def forward(self, x):
        # SwiGLU 激活机制
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class GemmaAttention(nn.Module):
    """Multi-Query Attention (MQA) 实现"""
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        
        # Q 向量有多个头，但 K 和 V 只有 1 个头 (MQA 特性)
        self.q_proj = nn.Linear(config.width, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.width, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.width, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.width, bias=False)

    def forward(self, x):
        # 此处省略了复杂的 RoPE 旋转和 Scaled Dot-Product 计算
        # 逻辑：QKV 投影 -> 多头拆分 -> 注意力计算 -> 合并输出
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.o_proj(q) # 简化示意

class GemmaBlock(nn.Module):
    """单个 Transformer 层"""
    def __init__(self, config):
        super().__init__()
        self.attn = GemmaAttention(config)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = nn.LayerNorm(config.width) # 实际 Gemma 使用 RMSNorm
        self.post_attention_layernorm = nn.LayerNorm(config.width)

    def forward(self, x):
        # 残差结构
        x = x + self.attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

# --- 2. 实例化并统计参数 ---

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 定义配置
gemma_2b_cfg = GemmaConfig(2048, 18, 16384, 8, 1, 256)
gemma_300m_cfg = GemmaConfig(1024, 18, 4096, 8, 1, 256)

# 实例化单层
block_2b = GemmaBlock(gemma_2b_cfg)
block_300m = GemmaBlock(gemma_300m_cfg)

print(f"Gemma 2B 单层参数量: {count_parameters(block_2b) / 1e6:.2f} M")
print(f"Gemma 300M 单层参数量: {count_parameters(block_300m) / 1e6:.2f} M")
```