```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class KVCacheSelfAttention(nn.Module):
    """
    带KV缓存的自注意力模块
    支持增量推理，适用于生成式模型
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        bias: bool = True,
        flash_attention: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.flash_attention = flash_attention
        
        # 检查维度是否能被头数整除
        assert self.head_dim * num_heads == dim, "dim必须能被num_heads整除"
        
        # 投影层
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        
        # dropout层
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        is_causal: bool = True,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播
        
        参数:
            x: 输入张量 [batch_size, seq_len, dim]
            kv_cache: 之前的KV缓存 (key, value)
            attention_mask: 注意力掩码
            use_cache: 是否返回KV缓存
            is_causal: 是否使用因果注意力掩码
            past_key_values: 之前的键值对（用于兼容一些API）
            
        返回:
            output: 输出张量 [batch_size, seq_len, dim]
            kv_cache: 更新后的KV缓存 (key, value)
        """
        batch_size, seq_len, _ = x.shape
        
        # 如果提供了past_key_values，转换为kv_cache格式
        if past_key_values is not None and kv_cache is None:
            kv_cache = past_key_values
        
        # 计算Q, K, V
        q = self.q_proj(x)  # [batch_size, seq_len, dim]
        k = self.k_proj(x)  # [batch_size, seq_len, dim]
        v = self.v_proj(x)  # [batch_size, seq_len, dim]
        
        # 重塑为多头格式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 如果使用KV缓存，拼接之前的K和V
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            # k_cache: [batch_size, num_heads, cache_seq_len, head_dim]
            # v_cache: [batch_size, num_heads, cache_seq_len, head_dim]
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        
        # 准备返回的缓存
        new_kv_cache = (k, v) if use_cache else None
        
        # 计算注意力分数
        if self.flash_attention and torch.cuda.is_available() and is_causal:
            # 使用Flash Attention（如果可用）
            output = self._flash_attention(q, k, v, attention_mask, is_causal)
        else:
            # 标准注意力实现
            output = self._standard_attention(q, k, v, attention_mask, is_causal)
        
        # 重塑输出
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        # 输出投影和dropout
        output = self.out_proj(output)
        output = self.resid_dropout(output)
        
        return output, new_kv_cache
    
    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        is_causal: bool
    ) -> torch.Tensor:
        """标准注意力实现"""
        batch_size, num_heads, q_seq_len, head_dim = q.shape
        _, _, k_seq_len, _ = k.shape
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 应用因果掩码（如果需要）
        if is_causal and attention_mask is None:
            # 创建因果掩码
            causal_mask = torch.tril(
                torch.ones((q_seq_len, k_seq_len), dtype=torch.bool, device=q.device)
            )
            causal_mask = causal_mask.view(1, 1, q_seq_len, k_seq_len)
            attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))
        
        # 应用外部提供的注意力掩码
        if attention_mask is not None:
            # attention_mask: [batch_size, 1, q_seq_len, k_seq_len] 或 [batch_size, q_seq_len, k_seq_len]
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attn_scores = attn_scores + attention_mask
        
        # 应用softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        
        # 计算输出
        output = torch.matmul(attn_probs, v)
        
        return output
    
    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        is_causal: bool
    ) -> torch.Tensor:
        """Flash Attention实现（需要PyTorch 2.0+）"""
        try:
            import torch.nn.functional as F
            
            # 使用scaled_dot_product_attention
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal
            )
            return output
        except ImportError:
            # 如果Flash Attention不可用，回退到标准注意力
            return self._standard_attention(q, k, v, attention_mask, is_causal)


class KVCacheManager:
    """
    KV缓存管理器，用于管理多个层的KV缓存
    """
    
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.cache = [None] * num_layers
    
    def init_cache(self, batch_size: int, max_length: int, device: torch.device):
        """初始化缓存"""
        self.cache = [None] * self.num_layers
    
    def update_cache(self, layer_idx: int, new_cache: Tuple[torch.Tensor, torch.Tensor]):
        """更新指定层的缓存"""
        self.cache[layer_idx] = new_cache
    
    def get_cache(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """获取指定层的缓存"""
        if layer_idx < len(self.cache):
            return self.cache[layer_idx]
        return None
    
    def clear_cache(self):
        """清空所有缓存"""
        self.cache = [None] * self.num_layers


# 使用示例
def example_usage():
    # 配置参数
    batch_size = 2
    seq_len = 10
    dim = 512
    num_heads = 8
    num_layers = 6
    
    # 创建模型
    attn_layer = KVCacheSelfAttention(dim=dim, num_heads=num_heads)
    cache_manager = KVCacheManager(num_layers=num_layers)
    
    # 模拟输入
    x = torch.randn(batch_size, seq_len, dim)
    
    print("=== 第一次前向传播（无缓存）===")
    output1, kv_cache = attn_layer(x, use_cache=True)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output1.shape}")
    print(f"KV缓存形状: k={kv_cache[0].shape}, v={kv_cache[1].shape}")
    
    # 保存缓存
    cache_manager.update_cache(0, kv_cache)
    
    print("\n=== 第二次前向传播（有缓存，增量推理）===")
    # 模拟增量输入（通常是单个token）
    x_incremental = torch.randn(batch_size, 1, dim)
    
    # 获取之前的缓存
    past_kv = cache_manager.get_cache(0)
    
    output2, new_kv_cache = attn_layer(
        x_incremental,
        kv_cache=past_kv,
        use_cache=True,
        is_causal=True
    )
    
    print(f"增量输入形状: {x_incremental.shape}")
    print(f"增量输出形状: {output2.shape}")
    print(f"新KV缓存形状: k={new_kv_cache[0].shape}, v={new_kv_cache[1].shape}")
    
    # 验证缓存是否正确拼接
    print(f"\n=== 缓存验证 ===")
    print(f"原始K形状: {past_kv[0].shape if past_kv else '无'}")
    print(f"新增K形状: {x_incremental.shape}")
    print(f"拼接后K形状: {new_kv_cache[0].shape}")
    print(f"缓存序列长度增加: {new_kv_cache[0].shape[2] - (past_kv[0].shape[2] if past_kv else 0)}")


# 测试函数
def test_kv_cache():
    """测试KV缓存功能"""
    torch.manual_seed(42)
    
    # 简单测试
    dim = 64
    num_heads = 4
    batch_size = 1
    seq_len1 = 5
    seq_len2 = 3
    
    attn = KVCacheSelfAttention(dim=dim, num_heads=num_heads)
    
    # 第一次前向传播
    x1 = torch.randn(batch_size, seq_len1, dim)
    output1, kv_cache1 = attn(x1, use_cache=True)
    
    # 第二次前向传播（使用缓存）
    x2 = torch.randn(batch_size, seq_len2, dim)
    output2, kv_cache2 = attn(x2, kv_cache=kv_cache1, use_cache=True)
    
    # 一次性处理所有token（用于验证）
    x_all = torch.cat([x1, x2], dim=1)
    output_all, _ = attn(x_all, use_cache=False)
    
    # 检查结果是否一致
    output2_from_all = output_all[:, seq_len1:, :]
    
    # 允许小的数值误差
    tolerance = 1e-5
    if torch.allclose(output2, output2_from_all, rtol=tolerance, atol=tolerance):
        print("✓ KV缓存测试通过：增量推理与一次性计算结果一致")
        return True
    else:
        print("✗ KV缓存测试失败：结果不一致")
        max_diff = torch.max(torch.abs(output2 - output2_from_all))
        print(f"最大差异: {max_diff.item()}")
        return False


if __name__ == "__main__":
    print("带KV缓存的自注意力实现")
    print("=" * 50)
    
    # 运行测试
    test_kv_cache()
    
    print("\n" + "=" * 50)
    print("使用示例：")
    example_usage()
```