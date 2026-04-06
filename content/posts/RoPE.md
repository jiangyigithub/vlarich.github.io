## https://yuanchaofa.com/post/hands-on-rope-position-embedding
```python
class MultiHeadAttentionWithRoPE(torch.nn.Module):
    """
    带 RoPE 的多头注意力
    """
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 4096):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.k_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.v_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.o_proj = torch.nn.Linear(d_model, d_model, bias=False)

        # RoPE
        self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: 输入，shape (batch, seq_len, d_model)
            mask: 注意力掩码，shape (seq_len, seq_len)
        """
        batch, seq_len, _ = x.shape

        # 线性投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 重塑为多头形式
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim)

        # 应用 RoPE（只对 Q 和 K）
        q, k = self.rope(q, k) #rope

        # 转置用于矩阵乘法：(batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 计算注意力分数
        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)

        # 加权求和
        attn_output = torch.matmul(attn_probs, v)

        # 重塑并输出投影
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, self.d_model)
        output = self.o_proj(attn_output)

        return output


# 测试
mha = MultiHeadAttentionWithRoPE(d_model=512, num_heads=8)
x = torch.randn(2, 128, 512)
output = mha(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

```python
class RotaryPositionEmbedding(torch.nn.Module):
    """
    完整的 RoPE 实现
    """
    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        """
        Args:
            dim: 每个注意力头的维度
            max_seq_len: 最大序列长度
            theta: 基础频率参数
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # 预计算并缓存 sin/cos 值
        cos, sin = get_rotary_embedding(dim, max_seq_len, theta)
        self.register_buffer('cos_cached', cos)
        self.register_buffer('sin_cached', sin)

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor = None):
        """
        对 Query 和 Key 应用 RoPE

        Args:
            q: Query，shape (batch, seq_len, num_heads, head_dim)
            k: Key，shape (batch, seq_len, num_heads, head_dim)
            positions: 位置索引，默认为 [0, 1, 2, ..., seq_len-1]

        Returns:
            q_rot, k_rot: 旋转后的 Query 和 Key
        """
        seq_len = q.shape[1]

        # 获取当前序列长度的 cos/sin
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]

        # 应用旋转
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

        return q_rot, k_rot


# 测试
rope = RotaryPositionEmbedding(dim=64, max_seq_len=4096)

# 模拟输入
batch_size = 2
seq_len = 128
num_heads = 8
head_dim = 64

q = torch.randn(batch_size, seq_len, num_heads, head_dim)
k = torch.randn(batch_size, seq_len, num_heads, head_dim)

q_rot, k_rot = rope(q, k)
print(f"Q_rot shape: {q_rot.shape}")
print(f"K_rot shape: {k_rot.shape}")
```